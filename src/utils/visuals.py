import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, Optional
import cv2

matplotlib.use("Agg")


class DPTVisualizer:
    """Visualization utilities for DPT outputs."""

    @staticmethod
    def visualize_depth(
        depth: np.ndarray,
        colormap: str = "magma",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> np.ndarray:
        """
        Visualize depth map.

        Args:
            depth: Depth map (H, W) or (1, H, W)
            colormap: Matplotlib colormap name
            vmin, vmax: Value range for normalization

        Returns:
            RGB visualization (H, W, 3) in range [0, 255]
        """
        if depth.ndim == 3:
            depth = depth[0]  # Remove channel dimension

        # Normalize depth
        if vmin is None:
            vmin = depth.min()
        if vmax is None:
            vmax = depth.max()

        depth_normalized = (depth - vmin) / (vmax - vmin + 1e-6)
        depth_normalized = np.clip(depth_normalized, 0, 1)

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        depth_colored = cmap(depth_normalized)[:, :, :3]  # Remove alpha
        depth_colored = (depth_colored * 255).astype(np.uint8)

        return depth_colored

    @staticmethod
    def visualize_vector_map(vec_map: np.ndarray) -> np.ndarray:
        """
        Visualize a 3-channel vector map as RGB.
        Args:
            vec_map: (3, H, W)
        Returns:
            RGB visualization (H, W, 3)
        """
        if vec_map.ndim == 3:
            vec_map = vec_map.transpose(1, 2, 0)  # H, W, 3

        # Normalize to [0, 1] for visualization
        v_min = vec_map.min()
        v_max = vec_map.max()
        if v_max - v_min > 1e-6:
            vec_vis = (vec_map - v_min) / (v_max - v_min)
        else:
            vec_vis = np.zeros_like(vec_map)

        return (vec_vis * 255).astype(np.uint8)

    @staticmethod
    def create_comparison_grid(
        image: np.ndarray,
        predictions: Dict[str, np.ndarray],
        metrics=None,
        ground_truth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create a grid visualization of all predictions.

        Args:
            image: Original RGB image (H, W, 3)
            predictions: Dictionary of predictions
            metrics: Dictionary of metric values
            ground_truth: Optional Ground Truth RGB image (H, W, 3)

        Returns:
            Grid visualization
        """
        # Prepare visualizations
        vis_items = [("Input", image)]
        
        if ground_truth is not None:
             vis_items.append(("Ground Truth", ground_truth))

        if "depth" in predictions:
            depth_vis = DPTVisualizer.visualize_depth(predictions["depth"])
            vis_items.append(("Depth", depth_vis))

        if "sharp_image" in predictions:
            sharp = predictions["sharp_image"]
            if sharp.ndim == 3:
                sharp = sharp.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            sharp = (sharp * 255).astype(np.uint8)
            vis_items.append(("Predicted Sharp", sharp))

        if "blur_image" in predictions:
            blur = predictions["blur_image"]
            if blur.ndim == 3:
                blur = blur.transpose(1, 2, 0)
            blur = (blur * 255).astype(np.uint8)
            vis_items.append(("Blur Image", blur))

        if "motion" in predictions:
            motion = predictions["motion"]
            # motion is (6, H, W)
            # 0:3 -> Linear Velocity
            # 3:6 -> Angular Velocity
            linear = motion[0:3, :, :]
            angular = motion[3:6, :, :]

            vis_items.append(("Linear Velocity", DPTVisualizer.visualize_vector_map(linear)))
            vis_items.append(("Angular Velocity", DPTVisualizer.visualize_vector_map(angular)))

        title_str = ""
        if metrics is not None:
            metric_str = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            title_str = "\n".join(metric_str)

        # Create grid
        n_items = len(vis_items)
        n_cols = 2
        n_rows = (n_items + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_rows > 1 and n_cols == 1:
             axes = axes.reshape(-1, 1)

        if title_str:
            fig.suptitle(title_str, fontsize=14, y=0.98)

        # Handle single subplot case (axes is not array if 1x1, but we forced reshape or grid size >= 2 usually)
        # With n_cols=2, we always have array unless n_items=1 which is unlikely (Input + something).
        
        # Flatten axes for easy iteration if it's a grid
        axes_flat = axes.flatten()

        for idx, (title, vis) in enumerate(vis_items):
            axes_flat[idx].imshow(vis)
            axes_flat[idx].set_title(title, fontsize=14)
            axes_flat[idx].axis("off")

        # Hide unused subplots
        for idx in range(n_items, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Convert to numpy array
        fig.canvas.draw()
        grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return grid


def log_visualizations(
    model, writer, epoch, blur_imgs, sharp_imgs, metrics, device
):
    """
    Custom visualization logger for MDPhysics model.
    Handles dictionary output and logs depth + motion field.
    """
    model.eval()
    with torch.no_grad():
        blur_imgs = blur_imgs.to(device)
        sharp_imgs = sharp_imgs.to(device)

        outputs = model(blur_imgs)

        # Loop over each image in the batch
        for i in range(blur_imgs.size(0)):
            # Prepare inputs for visualization (convert to numpy HWC)
            blur_np = blur_imgs[i].permute(1, 2, 0).cpu().numpy()
            blur_np = (blur_np * 255).astype(np.uint8)
            
            sharp_np = sharp_imgs[i].permute(1, 2, 0).cpu().numpy()
            sharp_np = (sharp_np * 255).astype(np.uint8)

            # Prepare predictions dict
            sample_preds = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    sample_preds[k] = v[i].cpu().numpy()

            # Calculate metrics for this image (using Tensors)
            img_metrics = {}
            if "sharp_image" in outputs:
                pred_sharp_tensor = outputs["sharp_image"][i]
                target_sharp_tensor = sharp_imgs[i]
                
                # Clamp for metrics
                pred_sharp_clamped = torch.clamp(pred_sharp_tensor, 0, 1)

                for name, metric in metrics.items():
                    metric.reset()
                    metric.update(pred_sharp_clamped.unsqueeze(0), target_sharp_tensor.unsqueeze(0))
                    img_metrics[name] = metric.compute().item()

            writer.add_image(
                f"visuals/{i}",
                DPTVisualizer.create_comparison_grid(
                    image=blur_np, 
                    predictions=sample_preds, 
                    metrics=img_metrics,
                    ground_truth=sharp_np
                ),
                epoch,
                dataformats="HWC",
            )
