"""Training-only hooks for the official microsoft/Lens DiT.

We do not replace Lens weights, config.json, or model classes. The only change
here is an optional ``forward`` wrapper so block-wise activation checkpointing
works on 16GB GPUs. ``LensTransformer2DModel.__init__`` is left untouched so
Diffusers loads Hub config fields without spurious warnings.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch


def apply_lens_training_patches() -> None:
    """Install the forward checkpointing hook (safe to call after ``from_pretrained``)."""
    _patch_lens_transformer_forward_checkpointing()


def _patch_lens_transformer_forward_checkpointing() -> None:
    try:
        from lens.transformer import LensTransformer2DModel
    except ImportError:
        return

    if getattr(LensTransformer2DModel, "_lens_trainer_gc_patch", False):
        return
    LensTransformer2DModel._lens_trainer_gc_patch = True

    original_forward = LensTransformer2DModel.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_hidden_states_mask: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: List[Tuple[int, int, int]],
        attention_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        if not (torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False)):
            return original_forward(
                self,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                timestep,
                img_shapes,
                attention_kwargs=attention_kwargs,
            )

        bsz, img_len, _ = hidden_states.shape
        if self.multi_layer_encoder_feature:
            if not isinstance(encoder_hidden_states, (list, tuple)):
                raise ValueError(
                    "multi_layer_encoder_feature=True expects a list of "
                    "per-layer text tensors."
                )
            if len(encoder_hidden_states) != len(self.selected_layer_index):
                raise ValueError(
                    f"Expected {len(self.selected_layer_index)} text feature "
                    f"layers, got {len(encoder_hidden_states)}."
                )
            text_seq_len = encoder_hidden_states[0].shape[1]
            for i, feat in enumerate(encoder_hidden_states):
                if feat.shape[0] != bsz:
                    raise ValueError(
                        f"Text feature layer {i} batch size {feat.shape[0]} "
                        f"does not match hidden_states batch size {bsz}."
                    )
                if feat.shape[1] != text_seq_len:
                    raise ValueError(
                        f"Text feature layer {i} sequence length {feat.shape[1]} "
                        f"does not match layer 0 length {text_seq_len}."
                    )
        else:
            if not isinstance(encoder_hidden_states, torch.Tensor):
                raise ValueError(
                    "multi_layer_encoder_feature=False expects a single text "
                    "feature tensor."
                )
            if encoder_hidden_states.shape[0] != bsz:
                raise ValueError(
                    f"Text feature batch size {encoder_hidden_states.shape[0]} "
                    f"does not match hidden_states batch size {bsz}."
                )
            text_seq_len = encoder_hidden_states.shape[1]
        if encoder_hidden_states_mask.shape != (bsz, text_seq_len):
            raise ValueError(
                "encoder_hidden_states_mask must have shape "
                f"{(bsz, text_seq_len)}, got {tuple(encoder_hidden_states_mask.shape)}."
            )
        attention_mask = self._build_joint_attention_mask(
            encoder_hidden_states_mask, img_len
        )

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.multi_layer_encoder_feature:
            normed = [
                self.txt_norm[i](encoder_hidden_states[i])
                for i in range(len(self.selected_layer_index))
            ]
            encoder_hidden_states = torch.cat(normed, dim=-1)
        else:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)
        image_rotary_emb = self.pos_embed(
            img_shapes, [text_seq_len], device=hidden_states.device
        )

        for block in self.transformer_blocks:

            def _run_block(
                blk,
                hs,
                enc,
                tb,
                rot,
                am,
            ):
                return blk(
                    hidden_states=hs,
                    encoder_hidden_states=enc,
                    temb=tb,
                    image_rotary_emb=rot,
                    attention_mask=am,
                )

            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                _run_block,
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_mask,
                use_reentrant=False,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)

    LensTransformer2DModel.forward = forward


def prepare_lens_transformer_for_load(transformer: torch.nn.Module) -> None:
    """After Hub load: ensure checkpointing flag exists and is off until training enables it."""
    apply_lens_training_patches()
    transformer.gradient_checkpointing = False


def enable_lens_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Turn on block-wise activation checkpointing on a (PEFT-wrapped) Lens DiT."""
    apply_lens_training_patches()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "transformer_blocks"):
        raise ValueError(
            f"Expected LensTransformer2DModel, got {type(base).__name__}"
        )
    base.gradient_checkpointing = True
