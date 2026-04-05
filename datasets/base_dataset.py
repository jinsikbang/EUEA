import os
import json
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from omegaconf import DictConfig


class BaseEmbodiedDataset(Dataset):
    """Base class for embodied scene-understanding datasets.

    Subclasses should implement ``_load_annotations`` and
    ``__getitem__``.
    """

    DEFAULT_IMAGE_SIZE = 336

    def __init__(self, cfg: DictConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.data_root = cfg.data_root
        self.image_size = cfg.get("image_size", self.DEFAULT_IMAGE_SIZE)

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.annotations = self._load_annotations()

    def _load_annotations(self) -> list:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.image_transform(img)

    def _load_pointcloud(
        self, pc_path: str, num_points: int = 4096
    ) -> torch.Tensor:
        """Load and sample a point cloud from a .npy or .bin file.

        Args:
            pc_path: Absolute path to the point cloud file.
            num_points: Number of points to sample.

        Returns:
            Tensor of shape (num_points, 6) containing (x, y, z, r, g, b).
        """
        if pc_path.endswith(".npy"):
            pc = np.load(pc_path)
        elif pc_path.endswith(".bin"):
            pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 6)
        else:
            raise ValueError(f"Unsupported point cloud format: {pc_path}")

        if len(pc) > num_points:
            indices = np.random.choice(len(pc), num_points, replace=False)
            pc = pc[indices]
        elif len(pc) < num_points:
            pad = num_points - len(pc)
            pc = np.concatenate([pc, pc[:pad]], axis=0)

        return torch.from_numpy(pc).float()
