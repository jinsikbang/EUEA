import os
import json
import torch
from omegaconf import DictConfig

from .base_dataset import BaseEmbodiedDataset


class EmbodiedScanDataset(BaseEmbodiedDataset):
    """Dataset wrapper for EmbodiedScan (Wang et al., 2023).

    EmbodiedScan is a large-scale 3D scan dataset with multi-modal
    annotations for embodied scene understanding tasks, including
    3D visual grounding and 3D dense captioning.

    Dataset structure expected under ``cfg.data_root``::

        embodiedscan/
        ├── embodiedscan_infos_train.pkl
        ├── embodiedscan_infos_val.pkl
        ├── embodiedscan_infos_test.pkl
        ├── images/
        │   └── <scene_id>/<frame_id>.jpg
        └── pointclouds/
            └── <scene_id>.npy
    """

    TASK_TYPES = ("grounding", "captioning", "qa")

    def __init__(self, cfg: DictConfig, split: str = "train"):
        self.task = cfg.get("task", "qa")
        assert self.task in self.TASK_TYPES, (
            f"Unknown task '{self.task}'. Choose from {self.TASK_TYPES}."
        )
        super().__init__(cfg, split)

    def _load_annotations(self) -> list:
        ann_file = os.path.join(
            self.data_root,
            "embodiedscan",
            f"embodiedscan_infos_{self.split}.json",
        )
        with open(ann_file) as f:
            raw = json.load(f)
        return raw.get(self.task, raw)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        scene_id = ann["scene_id"]

        # Load primary view image
        image_path = os.path.join(
            self.data_root, "embodiedscan", "images",
            scene_id, f"{ann['frame_id']}.jpg",
        )
        image = self._load_image(image_path)

        # Load point cloud
        pc_path = os.path.join(
            self.data_root, "embodiedscan", "pointclouds",
            f"{scene_id}.npy",
        )
        point_cloud = self._load_pointcloud(
            pc_path, self.cfg.get("num_points", 4096)
        )

        sample = {
            "image": image,
            "point_cloud": point_cloud,
            "scene_id": scene_id,
            "question": ann.get("question", "Describe the scene."),
            "answer": ann.get("answer", ""),
        }

        if self.task == "grounding":
            sample["bbox_3d"] = torch.tensor(ann["bbox_3d"], dtype=torch.float32)

        return sample
