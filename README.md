# LensTrainer-LoboForge

Config-driven LoRA trainer for [Microsoft Lens](https://github.com/microsoft/Lens) text-to-image models. Train on **Lens-Base** with a YAML config and export ComfyUI-compatible LoRA weights.

**Repository:** https://github.com/loboforge/LensTrainer-LoboForge

## Requirements

- Python 3.11+
- CUDA GPU (24GB preset tested for A100/4090-class cards)
- Hugging Face account with access to gated models (`microsoft/Lens-Base`, GPT-OSS weights)

## Setup

```bash
cd /media/wrath/AI/LensTrainer-LoboForge
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` installs [microsoft/Lens](https://github.com/microsoft/Lens) from GitHub so `from lens import LensPipeline` resolves correctly.

### Hugging Face gated access

1. Accept the model license on the Hugging Face hub for `microsoft/Lens-Base`.
2. Log in locally:

```bash
huggingface-cli login
```

Lens-Base bundles GPT-OSS text encoder weights stored in MXFP4. On non-Hopper GPUs (A100, V100, RTX 4090), keep `disable_mxfp4: true` in your config so weights dequantize to bf16/fp16.

## Dataset layout

Place paired images and captions in a folder. Each image needs a sidecar caption file with the same basename:

```
/path/to/images/
├── photo001.jpg
├── photo001.txt
├── photo002.png
└── photo002.txt
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.

Caption files are plain UTF-8 text (one prompt per file). Use your trigger word consistently, e.g. `a photo of mytrigger person smiling`.

## Training

Edit `dataset.folder_path` (and optionally `job.output_dir`, `sample.trigger_word`) in a preset config, then run:

```bash
python train.py configs/train_lora_lens_base_24gb.yaml
```

Override any field from the CLI:

```bash
python train.py configs/train_lora_lens_base_24gb.yaml \
  --set dataset.folder_path=/data/my-subject \
  --set train.steps=500 \
  --set sample.trigger_word=mytrigger
```

### VRAM presets

| Config | Target GPU | Key toggles |
|--------|------------|-------------|
| `configs/train_lora_lens_base_24gb.yaml` | ~24GB | CPU offload, TE cache, latent cache, `disable_mxfp4`, grad checkpointing, AdamW 8-bit |
| `configs/train_lora_lens_base_48gb.yaml` | 48GB+ | No CPU offload, live text encoding, latent cache |

Outputs land in `job.output_dir`:

- `checkpoints/lora_step_*.safetensors` — periodic LoRA saves
- `lora_final.safetensors` — final weights
- `samples/` — preview PNGs during training
- `cache/` — precomputed latents and text embeddings (when enabled)

## Config reference

| Section | Field | Description |
|---------|-------|-------------|
| `job` | `name`, `output_dir` | Run name and output directory |
| `model` | `repo_id` | HF repo (`microsoft/Lens-Base`) |
| | `dtype` | `bfloat16`, `float16`, or `float32` |
| | `disable_mxfp4` | Dequantize GPT-OSS weights (required on most consumer/datacenter GPUs) |
| | `cpu_offload` | Diffusers sequential CPU offload |
| | `cache_text_embeddings` | Precompute 4-layer GPT-OSS features to disk |
| `dataset` | `folder_path`, `caption_ext`, `resolution` | Image folder, caption extension, square resolution |
| | `cache_latents` | Precompute VAE latents `[B, H×W, 128]` to disk |
| `lora` | `rank`, `alpha`, `target_modules` | PEFT LoRA hyperparameters |
| `train` | `steps`, `batch_size`, `learning_rate`, `optimizer` | Training schedule (`adamw8bit` or `adamw`) |
| | `gradient_checkpointing`, `timestep_type` | Memory vs. timestep sampling (`shift` or `uniform`) |
| | `save_every`, `sample_every` | Checkpoint and preview frequency |
| `sample` | `prompts`, `trigger_word`, `width`, `height`, `steps`, `cfg`, `seed` | Mid-training previews (`[trigger]` replaced) |

## Training details

- **Flow-match loss:** target = `noise - latents`, MSE on transformer velocity prediction
- **Timestep:** sampled on `[0, 1000)`, passed to the transformer as `timestep / 1000`
- **Latents:** sequence format `[B, H×W, 128]` via Flux2 VAE + BN normalize + 2×2 patchify (mirrors `LensPipeline._decode` in reverse)
- **Text:** GPT-OSS layers `(5, 11, 17, 23)`, chat template, `txt_offset=97` — four feature tensors plus bool mask

## ComfyUI export

LoRA checkpoints are saved with ComfyUI key names (`diffusion_model.*` instead of PEFT `base_model.model.*`). Load the `.safetensors` file in your existing ComfyUI Lens workflow alongside the base Lens checkpoint.

Example keys:

```
diffusion_model.transformer_blocks.0.img_qkv.lora_A.weight
diffusion_model.transformer_blocks.0.img_qkv.lora_B.weight
```

## License

MIT (trainer code). Microsoft Lens and Lens-Base weights are subject to their respective licenses on Hugging Face.
