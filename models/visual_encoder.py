import torch
import torch.nn as nn
import timm
from omegaconf import DictConfig


class VisualEncoder(nn.Module):
    """Visual encoder that processes RGB images and optional point clouds.

    The encoder uses a pretrained ViT backbone for image feature extraction
    and a lightweight PointNet-style module for point cloud processing.
    Both feature streams are fused via concatenation followed by a linear
    projection.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # 2D image encoder (ViT-based)
        image_model_name = cfg.get("image_model", "vit_large_patch14_clip_336.openai")
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=cfg.get("pretrained", True),
            num_classes=0,  # remove classification head
        )
        image_feat_dim = self.image_encoder.num_features

        # 3D point cloud encoder (PointNet-style)
        self.use_pointcloud = cfg.get("use_pointcloud", True)
        if self.use_pointcloud:
            pc_input_dim = cfg.get("pc_input_dim", 6)   # x,y,z,r,g,b
            pc_hidden_dim = cfg.get("pc_hidden_dim", 256)
            self.pc_encoder = nn.Sequential(
                nn.Linear(pc_input_dim, pc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(pc_hidden_dim, pc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(pc_hidden_dim, pc_hidden_dim),
            )
            fused_dim = image_feat_dim + pc_hidden_dim
        else:
            fused_dim = image_feat_dim

        output_dim = cfg.get("output_dim", 1024)
        self.fusion_proj = nn.Linear(fused_dim, output_dim)
        self.output_dim = output_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features from RGB images.

        Args:
            images: Tensor of shape (B, C, H, W) or (B, V, C, H, W) for
                    multi-view inputs.

        Returns:
            Image features of shape (B, num_patches, image_feat_dim).
        """
        if images.dim() == 5:
            B, V, C, H, W = images.shape
            images = images.view(B * V, C, H, W)
            feats = self.image_encoder.forward_features(images)
            # Remove CLS token and reshape
            feats = feats[:, 1:, :]  # (B*V, num_patches, D)
            _, P, D = feats.shape
            feats = feats.view(B, V * P, D)
        else:
            feats = self.image_encoder.forward_features(images)
            feats = feats[:, 1:, :]  # remove CLS token
        return feats

    def encode_pointcloud(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """Extract global features from point clouds via max-pooling.

        Args:
            point_clouds: Tensor of shape (B, N, 6).

        Returns:
            Point cloud features of shape (B, 1, pc_hidden_dim).
        """
        per_point_feats = self.pc_encoder(point_clouds)      # (B, N, D)
        global_feats, _ = per_point_feats.max(dim=1)         # (B, D)
        return global_feats.unsqueeze(1)                      # (B, 1, D)

    def forward(
        self,
        images: torch.Tensor,
        point_clouds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode all visual inputs and project to output dimension.

        Args:
            images: RGB tensor of shape (B, C, H, W).
            point_clouds: Optional point cloud tensor of shape (B, N, 6).

        Returns:
            Visual features of shape (B, num_tokens, output_dim).
        """
        image_feats = self.encode_image(images)  # (B, P, img_dim)

        if self.use_pointcloud and point_clouds is not None:
            pc_feats = self.encode_pointcloud(point_clouds)  # (B, 1, pc_dim)
            # Broadcast and concatenate along the feature dimension
            pc_feats = pc_feats.expand(-1, image_feats.shape[1], -1)
            visual_feats = torch.cat([image_feats, pc_feats], dim=-1)
        else:
            visual_feats = image_feats

        return self.fusion_proj(visual_feats)
