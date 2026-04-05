#!/usr/bin/env python3
"""Dataset download helper for EUEA.

Downloads and organises the datasets used in the EUEA paper:
  - EmbodiedScan
  - ScanQA
  - SQA3D

Usage
-----
Download all datasets:
    python scripts/download_datasets.py --data_root /data/euea

Download a specific dataset:
    python scripts/download_datasets.py --data_root /data/euea --datasets scanqa sqa3d

Show help:
    python scripts/download_datasets.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "embodiedscan": {
        "description": (
            "EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite "
            "Towards Embodied AI (Wang et al., NeurIPS 2024)"
        ),
        "homepage": "https://github.com/OpenRobotLab/EmbodiedScan",
        "instructions": (
            "EmbodiedScan data must be obtained via the official release.\n"
            "1. Visit https://github.com/OpenRobotLab/EmbodiedScan\n"
            "2. Follow the data preparation instructions to download ScanNet,\n"
            "   3RScan, and Matterport3D source scans.\n"
            "3. Download the EmbodiedScan annotation files from the HuggingFace Hub:\n"
            "   https://huggingface.co/datasets/OpenRobotLab/EmbodiedScan\n"
            "4. Place the downloaded data under: <data_root>/embodiedscan/\n"
            "   Expected layout:\n"
            "     embodiedscan/\n"
            "     ├── embodiedscan_infos_train.json\n"
            "     ├── embodiedscan_infos_val.json\n"
            "     ├── embodiedscan_infos_test.json\n"
            "     ├── images/<scene_id>/<frame_id>.jpg\n"
            "     └── pointclouds/<scene_id>.npy\n"
        ),
        "auto_download": False,
    },

    "scanqa": {
        "description": (
            "ScanQA: 3D Question Answering for Spatial Scene Understanding "
            "(Azuma et al., CVPR 2022)"
        ),
        "homepage": "https://github.com/ATR-DBI/ScanQA",
        "annotation_urls": {
            "ScanQA_v1.0_train.json": (
                "https://raw.githubusercontent.com/ATR-DBI/ScanQA/"
                "main/data/qa/ScanQA_v1.0_train.json"
            ),
            "ScanQA_v1.0_val.json": (
                "https://raw.githubusercontent.com/ATR-DBI/ScanQA/"
                "main/data/qa/ScanQA_v1.0_val.json"
            ),
            "ScanQA_v1.0_test_w_obj.json": (
                "https://raw.githubusercontent.com/ATR-DBI/ScanQA/"
                "main/data/qa/ScanQA_v1.0_test_w_obj.json"
            ),
        },
        "scan_instructions": (
            "ScanQA uses ScanNet RGB-D scans for images and point clouds.\n"
            "Request access and download ScanNet at: http://www.scan-net.org/\n"
            "Place the scans under: <data_root>/scanqa/scans/<scene_id>/\n"
            "  - color/<frame_id>.jpg  (RGB frames)\n"
            "  - <scene_id>_pointcloud.npy (pre-processed point cloud)\n"
            "To generate the .npy point cloud files from ScanNet meshes,\n"
            "use the provided preprocessing script:\n"
            "  python scripts/preprocess_scannet.py --data_root <data_root>\n"
        ),
        "auto_download": True,   # annotation JSONs can be auto-downloaded
    },

    "sqa3d": {
        "description": (
            "SQA3D: Situated Question Answering in 3D Scenes "
            "(Ma et al., ICLR 2023)"
        ),
        "homepage": "https://github.com/SilongYong/SQA3D",
        "annotation_urls": {
            "sqa_task/questions/v1_balanced_questions_train_scannetv2.json": (
                "https://raw.githubusercontent.com/SilongYong/SQA3D/"
                "main/data/v1_balanced_questions_train_scannetv2.json"
            ),
            "sqa_task/questions/v1_balanced_questions_val_scannetv2.json": (
                "https://raw.githubusercontent.com/SilongYong/SQA3D/"
                "main/data/v1_balanced_questions_val_scannetv2.json"
            ),
            "sqa_task/questions/v1_balanced_questions_test_scannetv2.json": (
                "https://raw.githubusercontent.com/SilongYong/SQA3D/"
                "main/data/v1_balanced_questions_test_scannetv2.json"
            ),
            "sqa_task/balanced/v1_balanced_sqa_annotations_train_scannetv2.json": (
                "https://raw.githubusercontent.com/SilongYong/SQA3D/"
                "main/data/v1_balanced_sqa_annotations_train_scannetv2.json"
            ),
            "sqa_task/balanced/v1_balanced_sqa_annotations_val_scannetv2.json": (
                "https://raw.githubusercontent.com/SilongYong/SQA3D/"
                "main/data/v1_balanced_sqa_annotations_val_scannetv2.json"
            ),
            "sqa_task/balanced/v1_balanced_sqa_annotations_test_scannetv2.json": (
                "https://raw.githubusercontent.com/SilongYong/SQA3D/"
                "main/data/v1_balanced_sqa_annotations_test_scannetv2.json"
            ),
        },
        "scan_instructions": (
            "SQA3D uses ScanNet RGB-D scans (same as ScanQA).\n"
            "Place the scans under: <data_root>/sqa3d/scans/<scene_id>/\n"
            "  - color/<frame_id>.jpg  (RGB frames)\n"
            "  - <scene_id>_pointcloud.npy (pre-processed point cloud)\n"
        ),
        "auto_download": True,
    },
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _download_file(url: str, dest_path: str, retries: int = 3) -> bool:
    """Download a single file with retry logic.

    Args:
        url: URL to download.
        dest_path: Local path to write the file to.
        retries: Number of download attempts.

    Returns:
        True if the download was successful, False otherwise.
    """
    import urllib.request
    import urllib.error

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            logger.info("[%d/%d] Downloading %s ...", attempt, retries, url)
            urllib.request.urlretrieve(url, dest_path)
            logger.info("Saved to %s", dest_path)
            return True
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)

    logger.error("Failed to download %s after %d attempts.", url, retries)
    return False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-dataset download logic
# ---------------------------------------------------------------------------

def download_scanqa(data_root: str) -> None:
    """Download ScanQA annotation files."""
    info = DATASETS["scanqa"]
    dest_base = os.path.join(data_root, "scanqa")
    logger.info("=== ScanQA ===")
    logger.info(info["description"])

    failed = []
    for filename, url in info["annotation_urls"].items():
        dest = os.path.join(dest_base, filename)
        if os.path.exists(dest):
            logger.info("Already exists, skipping: %s", dest)
            continue
        if not _download_file(url, dest):
            failed.append(filename)

    if failed:
        logger.error(
            "Some ScanQA annotation files failed to download: %s", failed
        )
    else:
        logger.info("ScanQA annotation files downloaded successfully.")

    logger.info("\n%s", info["scan_instructions"])


def download_sqa3d(data_root: str) -> None:
    """Download SQA3D annotation files."""
    info = DATASETS["sqa3d"]
    dest_base = os.path.join(data_root, "sqa3d")
    logger.info("=== SQA3D ===")
    logger.info(info["description"])

    failed = []
    for rel_path, url in info["annotation_urls"].items():
        dest = os.path.join(dest_base, rel_path)
        if os.path.exists(dest):
            logger.info("Already exists, skipping: %s", dest)
            continue
        if not _download_file(url, dest):
            failed.append(rel_path)

    if failed:
        logger.error(
            "Some SQA3D annotation files failed to download: %s", failed
        )
    else:
        logger.info("SQA3D annotation files downloaded successfully.")

    logger.info("\n%s", info["scan_instructions"])


def download_embodiedscan(data_root: str) -> None:
    """Print manual download instructions for EmbodiedScan."""
    info = DATASETS["embodiedscan"]
    logger.info("=== EmbodiedScan ===")
    logger.info(info["description"])
    logger.warning(
        "EmbodiedScan cannot be automatically downloaded. "
        "Please follow the instructions below:\n\n%s",
        info["instructions"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets used in the EUEA paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name}: {meta['description']}"
            for name, meta in DATASETS.items()
        ),
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root directory where datasets will be stored.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to download (default: all).",
    )
    parser.add_argument(
        "--info_only",
        action="store_true",
        help="Print dataset info and instructions without downloading.",
    )
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    logger.info("Data root: %s", os.path.abspath(args.data_root))

    if args.info_only:
        for name in args.datasets:
            info = DATASETS[name]
            print(f"\n{'=' * 60}")
            print(f"Dataset: {name}")
            print(f"Description: {info['description']}")
            print(f"Homepage: {info['homepage']}")
            if "instructions" in info:
                print(f"Instructions:\n{info['instructions']}")
        return

    download_fn = {
        "embodiedscan": download_embodiedscan,
        "scanqa": download_scanqa,
        "sqa3d": download_sqa3d,
    }

    for name in args.datasets:
        logger.info("\n%s\n", "=" * 60)
        download_fn[name](args.data_root)

    logger.info("\nAll requested downloads complete.")
    logger.info(
        "NOTE: Raw ScanNet scans are required for ScanQA and SQA3D. "
        "See the instructions printed above."
    )


if __name__ == "__main__":
    main()
