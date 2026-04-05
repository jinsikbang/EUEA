import os
import json
import torch
from omegaconf import DictConfig

from .base_dataset import BaseEmbodiedDataset


class ScanQADataset(BaseEmbodiedDataset):
    """Dataset wrapper for ScanQA (Azuma et al., 2022).

    ScanQA is a 3D question-answering dataset based on ScanNet scenes.
    Each sample consists of a scene point cloud, a natural language question
    about the 3D environment, and one or more free-form text answers.

    Dataset structure expected under ``cfg.data_root``::

        scanqa/
        ├── ScanQA_v1.0_train.json
        ├── ScanQA_v1.0_val.json
        ├── ScanQA_v1.0_test_w_obj.json
        └── scans/
            └── <scene_id>/
                ├── color/
                │   └── <frame_id>.jpg
                └── <scene_id>_vh_clean_2.ply  (or .npy)
    """

    SPLIT_FILES = {
        "train": "ScanQA_v1.0_train.json",
        "val": "ScanQA_v1.0_val.json",
        "test": "ScanQA_v1.0_test_w_obj.json",
    }

    def __init__(self, cfg: DictConfig, split: str = "train"):
        super().__init__(cfg, split)

    def _load_annotations(self) -> list:
        filename = self.SPLIT_FILES[self.split]
        ann_file = os.path.join(self.data_root, "scanqa", filename)
        with open(ann_file) as f:
            return json.load(f)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        scene_id = ann["scene_id"]

        # Load a representative frame image for the scene
        images_dir = os.path.join(
            self.data_root, "scanqa", "scans", scene_id, "color"
        )
        frame_files = sorted(os.listdir(images_dir))
        frame_path = os.path.join(images_dir, frame_files[len(frame_files) // 2])
        image = self._load_image(frame_path)

        # Load point cloud
        pc_path = os.path.join(
            self.data_root, "scanqa", "scans", scene_id,
            f"{scene_id}_pointcloud.npy",
        )
        point_cloud = self._load_pointcloud(
            pc_path, self.cfg.get("num_points", 4096)
        )

        # ScanQA may have multiple reference answers
        answers = ann.get("answers", [ann.get("answer", "")])

        return {
            "image": image,
            "point_cloud": point_cloud,
            "scene_id": scene_id,
            "question_id": ann.get("question_id", str(idx)),
            "question": ann["question"],
            "answers": answers,
            # Use first answer as the training target
            "answer": answers[0] if answers else "",
        }
