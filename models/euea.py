import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from omegaconf import DictConfig

from .visual_encoder import VisualEncoder
from .language_decoder import LanguageDecoder


class EUEA(nn.Module):
    """Environmental Understanding Vision-Language Model for Embodied Agent.

    EUEA integrates a 3D-aware visual encoder with a large language model
    decoder to enable environmental understanding in embodied AI settings.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Visual encoder (processes RGB images + point clouds)
        self.visual_encoder = VisualEncoder(cfg.model.visual_encoder)

        # Language decoder (LLM backbone with optional LoRA)
        self.language_decoder = LanguageDecoder(cfg.model.language_decoder)

        # Projection layer: maps visual features to language embedding space
        visual_dim = cfg.model.visual_encoder.output_dim
        lang_dim = cfg.model.language_decoder.hidden_dim
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, lang_dim),
            nn.GELU(),
            nn.Linear(lang_dim, lang_dim),
        )

        # Learnable query tokens for cross-attention
        num_query_tokens = cfg.model.get("num_query_tokens", 32)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, lang_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def encode_visual(
        self,
        images: torch.Tensor,
        point_clouds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode visual inputs (RGB frames + optional point clouds).

        Args:
            images: RGB image tensor of shape (B, C, H, W) or
                    multi-view tensor of shape (B, V, C, H, W).
            point_clouds: Optional point cloud tensor of shape (B, N, 6)
                          containing (x, y, z, r, g, b) values.

        Returns:
            Visual feature tensor of shape (B, num_query_tokens, lang_dim).
        """
        visual_feats = self.visual_encoder(images, point_clouds)
        projected = self.visual_projection(visual_feats)

        # Expand query tokens for the batch
        B = projected.shape[0]
        query_tokens = self.query_tokens.expand(B, -1, -1)

        # Simple mean-pooling fusion (cross-attention variant in full model)
        pooled = projected.mean(dim=1, keepdim=True)
        visual_tokens = query_tokens + pooled

        return visual_tokens

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        point_clouds: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass for EUEA.

        Args:
            images: RGB image tensor of shape (B, C, H, W).
            input_ids: Tokenized text input of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            labels: Ground-truth token IDs of shape (B, L) for training.
            point_clouds: Optional point cloud tensor of shape (B, N, 6).

        Returns:
            Dictionary with 'loss' (training) and/or 'logits' (inference).
        """
        visual_tokens = self.encode_visual(images, point_clouds)
        return self.language_decoder(
            visual_tokens=visual_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        point_clouds: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text conditioned on visual and language inputs.

        Args:
            images: RGB image tensor of shape (B, C, H, W).
            input_ids: Tokenized prompt of shape (B, L).
            attention_mask: Attention mask of shape (B, L).
            point_clouds: Optional point cloud tensor of shape (B, N, 6).
            max_new_tokens: Maximum number of tokens to generate.
            **generate_kwargs: Additional keyword arguments for HF generate.

        Returns:
            Generated token IDs of shape (B, L').
        """
        visual_tokens = self.encode_visual(images, point_clouds)
        return self.language_decoder.generate(
            visual_tokens=visual_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and config to *save_dir*."""
        import json
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            from omegaconf import OmegaConf
            json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=2)

    @classmethod
    def from_pretrained(cls, load_dir: str, cfg: DictConfig | None = None):
        """Load model weights from *load_dir*.

        Args:
            load_dir: Path to the directory containing ``model.pt`` and
                      optionally ``config.json``.
            cfg: Optional config override.  When *None* the config saved
                 alongside the weights is used.

        Returns:
            EUEA model instance loaded with saved weights.
        """
        import json
        import os
        if cfg is None:
            config_path = os.path.join(load_dir, "config.json")
            with open(config_path) as f:
                from omegaconf import OmegaConf
                cfg = OmegaConf.create(json.load(f))
        model = cls(cfg)
        state_dict = torch.load(
            os.path.join(load_dir, "model.pt"), map_location="cpu"
        )
        model.load_state_dict(state_dict)
        return model
