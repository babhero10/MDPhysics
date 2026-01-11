import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def visualize_depth(depth_tensor, cmap="magma"):
    """
    Converts a depth tensor to a colorized image.
    Following DA3, depth is typically scale-shift invariant[cite: 270].
    """
    # 1. Remove batch/channel dims and convert to numpy
    if torch.is_tensor(depth_tensor):
        depth = depth_tensor.detach().cpu().squeeze().numpy()
    else:
        depth = depth_tensor

    # 2. Normalize to [0, 1] for visualization
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    # 3. Apply Colormap
    cm = plt.get_cmap(cmap)
    depth_color = (cm(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    # Return as RGB image
    return depth_color


def visualize_motion_field(motion_field_tensor):
    """
    Converts a 3D motion field (Vx, Vy, Vz) into a colorized 2D flow-style image.
    Direction is mapped to Hue, and Magnitude is mapped to Value.
    """
    # 1. Process Tensor: [3, H, W] -> [H, W, 3]
    if torch.is_tensor(motion_field_tensor):
        field = motion_field_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    else:
        field = motion_field_tensor

    # 2. Extract 2D components for flow visualization (Vx, Vy)
    vx, vy = field[:, :, 0], field[:, :, 1]

    # 3. Calculate Magnitude and Angle
    mag, ang = cv2.cartToPolar(vx, vy)

    # 4. Create HSV image
    hsv = np.zeros((field.shape[0], field.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: Direction
    hsv[..., 1] = 255  # Saturation: Max
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: Speed

    # 5. Convert HSV to RGB
    motion_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return motion_rgb
