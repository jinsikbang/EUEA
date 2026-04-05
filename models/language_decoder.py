import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from omegaconf import DictConfig


class LanguageDecoder(nn.Module):
    """Language decoder backed by a pretrained causal LLM.

    Visual tokens are prepended to the text embedding sequence before being
    fed into the LLM, allowing the model to condition its outputs on the
    visual context.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        model_name = cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.hidden_dim = self.llm.config.hidden_size

        # Optionally apply LoRA for parameter-efficient fine-tuning
        if cfg.get("use_lora", True):
            lora_cfg = cfg.get("lora", {})
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("alpha", 32),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                target_modules=list(
                    lora_cfg.get("target_modules", ["q_proj", "v_proj"])
                ),
                bias="none",
            )
            self.llm = get_peft_model(self.llm, lora_config)

    def _prepend_visual_tokens(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Prepend visual token embeddings to the text embedding sequence."""
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        B, V_len = visual_tokens.shape[:2]
        visual_mask = torch.ones(B, V_len, dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        extended_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            ignore = torch.full(
                (B, V_len), fill_value=-100,
                dtype=labels.dtype, device=labels.device,
            )
            labels = torch.cat([ignore, labels], dim=1)

        return inputs_embeds, extended_mask, labels

    def forward(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Run a forward pass of the language decoder.

        Args:
            visual_tokens: Projected visual features (B, V, D).
            input_ids: Text token IDs (B, L).
            attention_mask: Text attention mask (B, L).
            labels: Ground-truth token IDs (B, L) with -100 for ignored
                    positions.

        Returns:
            Dictionary containing 'loss' (when *labels* is provided) and
            'logits'.
        """
        inputs_embeds, extended_mask, labels = self._prepend_visual_tokens(
            visual_tokens, input_ids, attention_mask, labels
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
        )
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    def generate(
        self,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text conditioned on visual and text inputs.

        Args:
            visual_tokens: Projected visual features (B, V, D).
            input_ids: Prompt token IDs (B, L).
            attention_mask: Prompt attention mask (B, L).
            max_new_tokens: Maximum number of tokens to generate.
            **generate_kwargs: Additional keyword arguments forwarded to
                ``model.generate``.

        Returns:
            Generated token IDs of shape (B, L').
        """
        inputs_embeds, extended_mask, _ = self._prepend_visual_tokens(
            visual_tokens, input_ids, attention_mask, labels=None
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
