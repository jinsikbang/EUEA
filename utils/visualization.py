from __future__ import annotations

import os
import numpy as np

try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def visualize_scene(
    point_cloud: np.ndarray,
    save_path: str | None = None,
    window_name: str = "EUEA Scene",
    show: bool = False,
) -> None:
    """Visualize a 3D point cloud scene.

    Args:
        point_cloud: NumPy array of shape (N, 6) with (x, y, z, r, g, b).
                     RGB values are expected in the range [0, 255].
        save_path: If provided, the screenshot is saved to this path.
        window_name: Title of the Open3D window.
        show: Whether to open an interactive window.
    """
    if not _O3D_AVAILABLE:
        raise ImportError(
            "open3d is required for scene visualization. "
            "Install it with: pip install open3d"
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    if point_cloud.shape[1] >= 6:
        colors = point_cloud[:, 3:6] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if show or save_path is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, visible=show)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        if save_path is not None:
            vis.capture_screen_image(save_path, do_render=True)
        vis.destroy_window()


def visualize_predictions(
    predictions: list[dict],
    save_dir: str,
    max_samples: int = 20,
) -> None:
    """Save a grid of question / prediction / answer triplets as images.

    Args:
        predictions: List of dicts, each with keys ``"question"``,
                     ``"prediction"``, and ``"answers"``.
        save_dir: Directory where the output PNG files are written.
        max_samples: Maximum number of samples to visualise.
    """
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for prediction visualization. "
            "Install it with: pip install matplotlib"
        )
    os.makedirs(save_dir, exist_ok=True)
    for i, sample in enumerate(predictions[:max_samples]):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        text = (
            f"Question:   {sample.get('question', '')}\n"
            f"Prediction: {sample.get('prediction', '')}\n"
            f"Reference:  {' | '.join(sample.get('answers', []))}"
        )
        ax.text(
            0.01, 0.5, text,
            fontsize=11, verticalalignment="center",
            transform=ax.transAxes, wrap=True,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
        )
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i:04d}.png"), dpi=100)
        plt.close(fig)
