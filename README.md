# LensTrainer-LoboForge

Config-driven LoRA trainer for [Microsoft Lens-Base](https://huggingface.co/microsoft/Lens-Base). Train subject/style LoRAs from a folder of images + captions, export ComfyUI-compatible weights, and preview samples during training.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (24GB preset tested target; 48GB preset for larger batches)
- Hugging Face account with access to gated models (`microsoft/Lens-Base`, GPT-OSS weights)

## Setup

```bash
cd /media/wrath/AI/LensTrainer-LoboForge
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The trainer depends on the official [microsoft/Lens](https://github.com/microsoft/Lens) package (installed from Git via `requirements.txt`). If you prefer a local clone instead:

```bash
git clone https://github.com/microsoft/Lens.git ../Lens
export PYTHONPATH="/path/to/Lens:${PYTHONPATH}"
pip install -r requirements.txt --no-deps  # install other deps only
```

Authenticate with Hugging Face (required for gated checkpoints):

```bash
huggingface-cli login
```

Accept the model license on the Hub for `microsoft/Lens-Base` before first run.

## Dataset layout

Place training pairs in a folder (default `./dataset`). Each image needs a sidecar caption file with the same basename:

```
dataset/
  img001.jpg
  img001.txt
  img002.png
  img002.txt
```

Supported images: `.jpg`, `.jpeg`, `.png`, `.webp`. Caption extension is configurable (`caption_ext`, default `txt`).

Include your trigger token in captions (e.g. `a photo of mytrigger person standing in a park`). Set `sample.trigger_word` in the config to match.

## Quick start

Edit `dataset.folder_path` in a preset (or pass overrides):

```bash
python train.py configs/train_lora_lens_base_24gb.yaml \
  --set dataset.folder_path=/path/to/your/dataset \
  --set sample.trigger_word=mytrigger
```

Outputs land in `job.output_dir`:

- `checkpoints/lora_step_*.safetensors` — periodic LoRA saves
- `lora_final.safetensors` — final weights (ComfyUI key format)
- `samples/` — preview PNGs during training
- `cache/` — optional latent and text-embedding caches
- `loss.json` — per-step training loss
- `config.resolved.json` — resolved config snapshot

## Config reference

| Section | Key | Description |
|---------|-----|-------------|
| **job** | `name`, `output_dir` | Run name and output directory |
| **model** | `repo_id` | HF model id (`microsoft/Lens-Base`) |
| | `dtype` | `bfloat16`, `float16`, or `float32` |
| | `disable_mxfp4` | Dequantize GPT-OSS TE to bf16 (required on non-Hopper GPUs) |
| | `cpu_offload` | Diffusers CPU offload (`text_encoder→transformer→vae`) |
| | `cache_text_embeddings` | Precompute GPT-OSS multi-layer features to disk |
| **dataset** | `folder_path`, `caption_ext`, `resolution` | Data root, caption extension, square training size |
| | `cache_latents` | Precompute VAE latents `[B, H×W, 128]` to disk |
| | `max_sequence_length` | GPT-OSS chat prompt max tokens |
| **lora** | `rank`, `alpha`, `dropout`, `target_modules` | PEFT LoRA on transformer Linear layers |
| **train** | `steps`, `batch_size`, `learning_rate`, `optimizer` | Training schedule (`adamw8bit` or `adamw`) |
| | `gradient_checkpointing`, `gradient_accumulation_steps` | VRAM / effective batch |
| | `timestep_type` | `shift` (logit-normal biased) or `uniform` |
| | `save_every`, `sample_every` | Checkpoint and preview interval |
| **sample** | `prompts`, `trigger_word`, `width`, `height`, `steps`, `cfg`, `seed` | Mid-training previews |

### VRAM presets

| File | Target | Notes |
|------|--------|-------|
| `configs/train_lora_lens_base_24gb.yaml` | ~24GB | Offload + TE/latent cache + mxfp4 off + grad checkpointing |
| `configs/train_lora_lens_base_48gb.yaml` | 48GB+ | No offload, batch 2, rank 32 |

Override any field at runtime:

```bash
python train.py configs/train_lora_lens_base_24gb.yaml --set train.steps=500 --set lora.rank=8
```

## Training details

- **Base model:** `microsoft/Lens-Base` (not RL/Turbo variants)
- **Loss:** Flow-matching MSE with target `noise - latents`
- **Timesteps:** Sampled 0…999, passed to the transformer as `timestep / 1000`
- **Text:** GPT-OSS layers `(5, 11, 17, 23)` with chat template and `txt_offset=97`
- **Latents:** FLUX.2 VAE encode → BN normalize → 2×2 patchify → sequence `[B, H×W, 128]`

Only the DiT transformer is trained; VAE and text encoder stay frozen.

## ComfyUI export

Saved LoRA files remap PEFT keys for ComfyUI Lens workflows:

- `base_model.model.*` → stripped
- optional `transformer.*` → `diffusion_model.*`
- final keys look like `diffusion_model.transformer_blocks.*.lora_A.weight`

Load `lora_final.safetensors` (or a step checkpoint) in your existing ComfyUI Lens workflow alongside the base Lens checkpoint.

## Hugging Face gated access

1. Create/login at https://huggingface.co
2. Request access to `microsoft/Lens-Base` on the model page
3. Run `huggingface-cli login` on the training machine
4. Ensure `disable_mxfp4: true` on Ampere/Ada GPUs (A100, RTX 30xx/40xx)

## License

MIT. Lens model weights and GPT-OSS components remain under their respective Microsoft / OpenAI licenses on Hugging Face.
