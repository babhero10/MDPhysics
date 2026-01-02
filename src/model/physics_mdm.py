import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class InteractionMatrixLayer(nn.Module):
    def __init__(self, height, width, focal_length):
        super().__init__()
        self.H = height
        self.W = width
        self.f = focal_length

        # Create invariant grid (u, v) centered at principal point
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.H), torch.arange(self.W), indexing="ij"
        )
        # Center coordinates (assuming principal point is at H/2, W/2)
        # We assume K is I matrix effectively, or we are working in pixel coordinates with f provided
        self.register_buffer("u", x_grid.float() - (self.W / 2), persistent=False)
        self.register_buffer("v", y_grid.float() - (self.H / 2), persistent=False)

    def forward(self, depth_map, velocity_vector):
        """
        depth_map: (B, 1, H, W) -> Learned Inverse Depth or Z
        velocity_vector: (B, 6) -> [Vx, Vy, Vz, wx, wy, wz]
        """
        B, _, H, W = depth_map.shape

        # Check if we need to regenerate the grid (for variable input sizes like validation)
        if self.u.shape[0] != H or self.u.shape[1] != W:
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=depth_map.device),
                torch.arange(W, device=depth_map.device),
                indexing="ij",
            )
            u = x_grid.float() - (W / 2)
            v = y_grid.float() - (H / 2)
        else:
            u, v = self.u, self.v

        f = self.f

        # Ensure Z is safe. The network predicts Inverse Depth (1/Z).
        inv_Z = depth_map

        # Unpack velocities (B, 1, 1, 1) for broadcasting
        Vx = velocity_vector[:, 0].view(B, 1, 1, 1)
        Vy = velocity_vector[:, 1].view(B, 1, 1, 1)
        Vz = velocity_vector[:, 2].view(B, 1, 1, 1)
        wx = velocity_vector[:, 3].view(B, 1, 1, 1)
        wy = velocity_vector[:, 4].view(B, 1, 1, 1)
        wz = velocity_vector[:, 5].view(B, 1, 1, 1)

        # --- BRANCH 1: TRANSLATIONAL FLOW ---
        # u_dot_trans = (-f/Z)Vx + (u/Z)Vz
        # v_dot_trans = (-f/Z)Vy + (v/Z)Vz

        u_dot_t = (-f * inv_Z) * Vx + (u * inv_Z) * Vz
        v_dot_t = (-f * inv_Z) * Vy + (v * inv_Z) * Vz
        flow_translation = torch.cat([u_dot_t, v_dot_t], dim=1)  # (B, 2, H, W)

        # --- BRANCH 2: ROTATIONAL FLOW ---
        # u_dot_rot = (uv/f)wx - (f + u^2/f)wy + v*wz
        u_dot_r = ((u * v) / f) * wx - (f + (u**2) / f) * wy + v * wz

        # v_dot_rot = (f + v^2/f)wx - (uv/f)wy - u*wz
        v_dot_r = (f + (v**2) / f) * wx - ((u * v) / f) * wy - u * wz

        flow_rotation = torch.cat([u_dot_r, v_dot_r], dim=1)  # (B, 2, H, W)

        return flow_translation, flow_rotation


class PhysicsInformedMDM(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, focal_length=1.0):
        super().__init__()
        self.H = height
        self.W = width
        self.f = focal_length
        self.in_channels = in_channels
        self.out_channels = out_channels

        # --- 1. The Parameter Estimator (The "Brain") ---

        # A) Velocity Head (Global motion: 6 values)
        # We process input features to find global velocity
        self.velocity_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(64, 6),  # [Vx, Vy, Vz, wx, wy, wz]
        )

        # B) Depth Head (Pixel-wise structure: HxW values)
        # Predict 'Inverse Depth' (1/Z)
        self.inverse_depth_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Forces output to be between 0 and 1
        )

        # 2. Physics Engine (Interaction Matrix)
        self.interaction_solver = InteractionMatrixLayer(height, width, focal_length)

        # 3. Deformable Convolutions
        # Standard Deformable Convolutions to apply the result
        # We use in_channels input and produce out_channels output
        self.dcn_trans = DeformConv2d(in_channels, out_channels, 3, padding=1)
        self.dcn_rot = DeformConv2d(in_channels, out_channels, 3, padding=1)

        # Layers to map 2-channel flow to DCN offsets (usually 18 channels for 3x3)
        # 2 channels (flow_u, flow_v) -> 18 channels (offsets for 9 kernel points)
        self.flow_to_offset_t = nn.Conv2d(2, 18, kernel_size=1)
        self.flow_to_offset_r = nn.Conv2d(2, 18, kernel_size=1)

        # Initial projection if needed (not strictly used if DCN handles it,
        # but the snippet had param prediction on features.
        # If in_channels is 3, we are predicting from raw image. This is fine.)

    def forward(self, x):
        """
        x: Input feature map (B, C, H, W)
        """
        # --- Step 1: Predict Learnable Parameters ---

        # Learn Velocity (B, 6)
        velocity = self.velocity_head(x)

        # Learn Inverse Depth (B, 1, H, W)
        inv_Z = self.inverse_depth_head(x)

        # --- Step 2: Calculate Optical Flow via Interaction Matrix ---
        flow_trans, flow_rot = self.interaction_solver(inv_Z, velocity)

        # --- Step 3: Apply to Deformable Conv ---

        # Convert physical flow to DCN offsets
        offsets_t = self.flow_to_offset_t(flow_trans)
        offsets_r = self.flow_to_offset_r(flow_rot)

        # Apply Deformable Convolutions
        # x is (B, in_channels, H, W) -> output is (B, out_channels, H, W)
        feat_t = self.dcn_trans(x, offsets_t)
        feat_r = self.dcn_rot(x, offsets_r)

        # Combine
        return feat_t + feat_r, inv_Z, velocity
