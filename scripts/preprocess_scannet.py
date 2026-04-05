#!/usr/bin/env python3
"""Preprocess ScanNet scans into the format expected by EUEA datasets.

This script reads ScanNet .ply mesh files and RGB frame images, then exports:
  - Per-scene point cloud files  (<scene_id>_pointcloud.npy)
    with shape (N, 6) containing (x, y, z, r, g, b) coordinates.

Usage
-----
    python scripts/preprocess_scannet.py \\
        --scannet_root /data/scannet \\
        --output_root /data/euea \\
        --datasets scanqa sqa3d \\
        --num_points 40000

The output is written to:
    <output_root>/{scanqa,sqa3d}/scans/<scene_id>/<scene_id>_pointcloud.npy
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_ply_as_array(ply_path: str) -> np.ndarray:
    """Load a ScanNet .ply file and return an (N, 6) float32 array.

    Requires the ``open3d`` package.

    Args:
        ply_path: Path to the .ply mesh file.

    Returns:
        NumPy array of shape (N, 6) with columns (x, y, z, r, g, b) where
        RGB values are in the range [0, 255].
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for preprocessing. "
            "Install it with: pip install open3d"
        )

    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=200_000)

    xyz = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).astype(np.float32)
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.float32)

    return np.concatenate([xyz, rgb], axis=1)


def preprocess_scene(
    scene_id: str,
    scannet_root: str,
    output_dir: str,
    num_points: int,
) -> bool:
    """Preprocess a single scene.

    Args:
        scene_id: ScanNet scene identifier (e.g. ``scene0000_00``).
        scannet_root: Root directory of the ScanNet dataset.
        output_dir: Directory where the .npy file is written.
        num_points: Number of points to sample for the output cloud.

    Returns:
        True if preprocessing was successful.
    """
    ply_path = os.path.join(
        scannet_root, "scans", scene_id,
        f"{scene_id}_vh_clean_2.ply",
    )
    if not os.path.exists(ply_path):
        logger.warning("PLY file not found, skipping: %s", ply_path)
        return False

    out_path = os.path.join(output_dir, scene_id, f"{scene_id}_pointcloud.npy")
    if os.path.exists(out_path):
        return True  # already processed

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pc = load_ply_as_array(ply_path)

    # Subsample
    if len(pc) > num_points:
        indices = np.random.choice(len(pc), num_points, replace=False)
        pc = pc[indices]

    np.save(out_path, pc.astype(np.float32))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ScanNet scans for EUEA."
    )
    parser.add_argument(
        "--scannet_root",
        required=True,
        help="Root directory of the ScanNet dataset.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory for EUEA datasets (e.g. /data/euea).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["scanqa", "sqa3d"],
        default=["scanqa", "sqa3d"],
        help="Which dataset sub-directories to populate (default: both).",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=40_000,
        help="Number of points per scene (default: 40000).",
    )
    args = parser.parse_args()

    # Discover all ScanNet scenes
    scans_root = os.path.join(args.scannet_root, "scans")
    scene_ids = sorted(
        d for d in os.listdir(scans_root)
        if os.path.isdir(os.path.join(scans_root, d))
        and d.startswith("scene")
    )
    logger.info("Found %d scenes in %s", len(scene_ids), scans_root)

    for dataset in args.datasets:
        output_dir = os.path.join(args.output_root, dataset, "scans")
        logger.info("Processing dataset: %s  ->  %s", dataset, output_dir)

        success = fail = 0
        for scene_id in tqdm(scene_ids, desc=dataset):
            ok = preprocess_scene(
                scene_id, args.scannet_root, output_dir, args.num_points
            )
            if ok:
                success += 1
            else:
                fail += 1

        logger.info(
            "%s: %d succeeded, %d failed", dataset, success, fail
        )

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
