# LensTrainer-LoboForge

Config-driven LoRA trainer for [Microsoft Lens-Base](https://huggingface.co/microsoft/Lens-Base). Train subject/style LoRAs from a folder of images + captions, export ComfyUI-compatible weights, and preview samples during training.

## Quickstart (TL;DR)

**Clone, install, train:**

```bash
git clone https://github.com/LoboForge/LoboForge-LensTrainer.git
cd LoboForge-LensTrainer
./scripts/quickstart.sh
source .venv/bin/activate
huggingface-cli login   # once — accept microsoft/Lens-Base on the Hub first

python train.py configs/train_lora_lens_base_24gb.yaml \
  --set dataset.folder_path=/path/to/your/dataset \
  --set sample.trigger_word=your_trigger \
  --set job.output_dir=./output/my-lora
```

**Or one curl (clone + venv + pip):**

```bash
curl -fsSL https://raw.githubusercontent.com/LoboForge/LoboForge-LensTrainer/main/scripts/quickstart.sh | bash
cd ~/LoboForge-LensTrainer && source .venv/bin/activate
# then huggingface-cli login and the python train.py command above
```

**Auto-train after install** (optional):

```bash
DATASET_PATH=/path/to/images TRIGGER_WORD=Willow OUTPUT_DIR=./output/willow \
  bash scripts/quickstart.sh
```

Done when you have `job.output_dir/lora_final.safetensors`. See [Dataset layout](#dataset-layout) and [VRAM](#vram--system-requirements) below before your first run.

## Requirements

- Python 3.11+
- **GPU:** NVIDIA CUDA — see [VRAM](#vram--system-requirements)
- **RAM:** 32GB+ system memory recommended (text cache precompute with default settings)
- Hugging Face account with access to gated models (`microsoft/Lens-Base`, GPT-OSS weights)

### VRAM & system requirements

Lens-Base’s DiT is ~**13.4GB** in bf16. This trainer keeps the text encoder and VAE off GPU during training by caching latents and captions to disk — that’s how 16GB cards work at all.

| | Minimum | Recommended |
|---|---------|-------------|
| **GPU VRAM** | **16GB** (RTX 4060 Ti 16GB, 5060 Ti, etc.) | **24GB** (RTX 3090/4090, A5000, …) |
| **System RAM** | **32GB** | **64GB** (faster / safer text precompute) |
| **Preset** | `configs/train_lora_lens_base_24gb.yaml` | same, or `48gb` if you have headroom |

**16GB path (what we test on):** `cpu_offload: true`, disk caches, `gradient_checkpointing: true`, `batch_size: 1`, `disable_mxfp4: true` (text precompute on CPU once). Training holds the full DiT on GPU — there is no smaller mode today without architectural changes.

**Below 16GB VRAM:** not supported for training with Lens-Base using this repo.

**48GB+:** use `configs/train_lora_lens_base_48gb.yaml` — no CPU offload, batch 2, rank 32.

Peak VRAM by phase (16GB preset):

| Phase | On GPU | Rough VRAM |
|-------|--------|------------|
| Training loop | DiT + LoRA only | ~14–16GB |
| Latent precompute | VAE only | ~2–4GB |
| Text precompute (default) | nothing (CPU) | ~0GB |
| Mid-run samples | DiT + VAE swap | ~14–16GB |

## Setup

Manual install (if you skip `quickstart.sh`):

```bash
git clone https://github.com/LoboForge/LoboForge-LensTrainer.git
cd LoboForge-LensTrainer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
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

Corrupt or unreadable images are skipped at startup with a log line (see `cache/manifest.json` → `skipped`).

## Example run

Character LoRA (LoboForge mascot Willow preset):

```bash
python train.py configs/train_lora_willow_24gb.yaml \
  --set model.repo_id=./models/Lens-Base \
  --set dataset.folder_path=/path/to/willow/images
```

Generic override example:

```bash
python train.py configs/train_lora_lens_base_24gb.yaml \
  --set dataset.folder_path=/path/to/your/dataset \
  --set sample.trigger_word=mytrigger
```

Outputs land in `job.output_dir`:

- `checkpoints/lora_step_*.safetensors` — periodic LoRA saves
- `lora_final.safetensors` — final weights (ComfyUI key format)
- `samples/` — preview PNGs (`step_000000_control_*` baseline on full prompt list, then `step_*_lora_*` during training)
- `cache/` — optional latent and text-embedding caches (see **Caching** below)
- `loss.json` — per-step training loss
- `config.resolved.json` — resolved config snapshot

## Caching (skip precompute on re-runs)

Before training starts, the trainer can build **disk caches** so the VAE and GPT-OSS text encoder are not needed in the training loop (same idea as [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit)). Both presets enable this by default:

- `dataset.cache_latents: true` — VAE encode each image once → `cache/latents/*.pt`
- `model.cache_text_embeddings: true` — GPT-OSS encode each caption once → `cache/text/*.pt`

### What runs where (first run only)

| Phase | Model | Typical device (24GB preset) | Rough time |
|-------|--------|------------------------------|------------|
| `Precomputing latents` | VAE only | GPU | ~seconds per image |
| `Precomputing text` | HF GPT-OSS text encoder | CPU when `disable_mxfp4: true` | ~8–15 min for ~80 images |

Text precompute on CPU is **not** because latents need CPU — latents already ran on GPU. It is because `disable_mxfp4: true` expands the Hub TE to full bf16 (~40GB in RAM), which does not fit whole on a 16GB GPU. That cost is **one-time per dataset/output folder**. Training then only reads cached tensors and trains the DiT LoRA.

### Re-use caches on later runs

Caches live under **`job.output_dir/cache/`**. If you start another run with the **same** `output_dir`, dataset, captions, and `resolution`, the trainer **skips** any step that already has a matching `.pt` file and goes straight to training.

```bash
# First run: pays precompute + training
python train.py configs/train_lora_lens_base_24gb.yaml \
  --set dataset.folder_path=/media/wrath/AI/Training/Ballet \
  --set job.output_dir=./output/lens-lora-ballet \
  --set sample.trigger_word=ballet

# Resume or tweak training only — reuses ./output/lens-lora-ballet/cache/
python train.py configs/train_lora_lens_base_24gb.yaml \
  --set dataset.folder_path=/media/wrath/AI/Training/Ballet \
  --set job.output_dir=./output/lens-lora-ballet \
  --set train.steps=3000 \
  --set lora.rank=16
```

Keep the same `job.output_dir` when you want to avoid re-encoding. Copy or symlink the whole output folder to train a new run from existing caches.

### When caches are rebuilt

Cache filenames are hashed from **image path**, **caption text**, and **`dataset.resolution`**. A file is recomputed only if its entry is missing. You get a new encode if you:

- change a caption or swap an image file
- change `dataset.resolution`
- point `job.output_dir` at a folder without caches
- delete `cache/latents/` or `cache/text/` (or the whole `cache/` tree)

To force a full refresh:

```bash
rm -rf ./output/lens-lora-ballet/cache
```

### Faster text precompute with MXFP4 (optional)

The 24GB preset defaults to `disable_mxfp4: true`, which dequantizes the Hub text encoder to bf16 and runs **text precompute on CPU** (~8–15 min for ~80 images). On **Blackwell (RTX 50xx)** and other GPUs with MXFP4 support, keep the quantized ~6GB encoder and run text precompute on **GPU** (often much faster):

```bash
pip install 'kernels>=0.12.0,<0.15' 'triton>=3.4.0'

python train.py configs/train_lora_lens_base_24gb.yaml \
  --set dataset.folder_path=/path/to/dataset \
  --set job.output_dir=./output/my-run \
  --set model.disable_mxfp4=false
```

| `disable_mxfp4` | Text encoder size | Text precompute (typical) |
|-----------------|-------------------|---------------------------|
| `true` (default) | ~40GB bf16 in RAM | CPU, slower |
| `false` | ~6GB MXFP4 | GPU layer offload, faster |

Use `disable_mxfp4: true` on Ampere/Ada (RTX 30xx/40xx, A100) where MXFP4 kernels are unavailable. Training behavior is the same either way once caches exist — only the one-time encode path changes.

If you switch `disable_mxfp4` after building caches, delete `cache/text/` (or the whole `cache/` folder) so captions are re-encoded with the new TE mode.

**Note:** `kernels>=0.15` is incompatible with `transformers` 5.9 — pin `kernels>=0.12.0,<0.15` (see Troubleshooting).

## Config reference

| Section | Key | Description |
|---------|-----|-------------|
| **job** | `name`, `output_dir` | Run name and output directory |
| **model** | `repo_id` | Hugging Face id (`microsoft/Lens-Base`) **or** path to a local HF-layout folder (see below) |
| | `dtype` | `bfloat16`, `float16`, or `float32` |
| | `disable_mxfp4` | `true`: dequantize TE to bf16 (CPU text precompute on 16GB). `false`: keep MXFP4 (~6GB, GPU text precompute on Blackwell) — use `--set model.disable_mxfp4=false` |
| | `cpu_offload` | Diffusers CPU offload (`text_encoder→transformer→vae`) |
| | `cache_text_embeddings` | Precompute GPT-OSS multi-layer features to disk |
| **dataset** | `folder_path`, `caption_ext`, `resolution` | Data root, caption extension, square training size |
| | `cache_latents` | Precompute VAE latents `[B, H×W, 128]` to disk |
| | `max_sequence_length` | GPT-OSS chat prompt max tokens |
| **lora** | `rank`, `alpha`, `dropout`, `target_modules` | PEFT LoRA on transformer Linear layers |
| **train** | `steps`, `batch_size`, `learning_rate`, `optimizer` | Training schedule (`adamw8bit` or `adamw`) |
| | `gradient_checkpointing`, `gradient_accumulation_steps` | VRAM / effective batch |
| | `timestep_type` | `shift` (logit-normal biased) or `uniform` |
| | `save_every`, `sample_every` | Checkpoint and preview interval (after early phase) |
| | `sample_every_early`, `sample_early_until` | Denser previews early (e.g. every 50 until step 500) |
| **sample** | `prompts` | Named prompt list (`[trigger]` → `trigger_word`); string or `{name, prompt}` |
| | `baseline_control` | Step-0 control grid on base model (same prompt list, `step_000000_control_*`) |
| | `walk_seed` | Increment seed per prompt so samples are not identical |
| | `trigger_word`, `width`, `height`, `steps`, `cfg`, `seed` | Preview generation settings |

### VRAM presets

| File | Target | Notes |
|------|--------|-------|
| `configs/train_lora_lens_base_24gb.yaml` | ~16–24GB | Offload + caches; `disable_mxfp4: true` by default. On RTX 50xx add `--set model.disable_mxfp4=false` for faster text precompute |
| `configs/train_lora_willow_24gb.yaml` | ~16–24GB | Character LoRA preset (LoboForge mascot Willow, 100-step smoke test) |
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

## Model weights (Hugging Face layout)

The trainer loads Lens via `LensPipeline.from_pretrained()`, which expects the **standard Hugging Face repo layout**:

```
models/Lens-Base/
  model_index.json
  text_encoder/
  tokenizer/
  transformer/
  vae/
  scheduler/
```

### Option A — download everything from the Hub (default)

Set `model.repo_id: microsoft/Lens-Base` (default). On first run, weights download into the Hugging Face cache.

### Option B — assemble locally, skip large re-downloads

If you already have compatible **single-file** DiT and/or VAE safetensors (e.g. from ComfyUI), use the assemble script to build the HF folder and only download the rest (text encoder, tokenizer, configs):

```bash
# Copy the example and edit paths on your machine
cp configs/assemble_weights.example.yaml configs/assemble_weights.local.yaml

python scripts/assemble_lens_repo.py --config configs/assemble_weights.local.yaml
```

Or pass paths on the CLI:

```bash
python scripts/assemble_lens_repo.py \
  --output ./models/Lens-Base \
  --transformer /path/to/lens_bf16.safetensors \
  --vae /path/to/flux2-vae.safetensors
```

Local files are **key-validated** before use. By default they are **symlinked** into the HF tree (`--copy` to duplicate instead). The **text encoder always comes from the Hub** — ComfyUI-packaged GPT-OSS files use a different format and are not supported.

Train against the assembled folder:

```bash
python train.py configs/train_lora_lens_base_24gb.yaml \
  --set model.repo_id=./models/Lens-Base
```

Check an existing folder:

```bash
python scripts/assemble_lens_repo.py --output ./models/Lens-Base --check
```

## Hugging Face gated access

1. Create/login at https://huggingface.co
2. Request access to `microsoft/Lens-Base` on the model page
3. Run `huggingface-cli login` on the training machine
4. **GPU choice:** `disable_mxfp4: true` on Ampere/Ada (RTX 30xx/40xx, A100). On Blackwell (RTX 50xx), use `--set model.disable_mxfp4=false` for faster text precompute (see **Caching**).

## Troubleshooting

### `Invalid handle. Cannot load symbol cublasLtGetVersion` (core dump)

This usually means the dynamic linker picked up the wrong `libcublasLt` — common when running from **Cursor’s integrated terminal**, which prepends AppImage paths to `LD_LIBRARY_PATH`.

`train.py` now sanitizes that automatically. If you still hit the error:

```bash
unset LD_LIBRARY_PATH
python train.py configs/train_lora_lens_base_24gb.yaml ...
```

Or run training from a normal system terminal (outside Cursor). Plain `python -c "import torch; ..."` can still work while Lens training fails, because the bad path only affects some CUDA ops after the full pipeline loads.

### CUDA OOM during `Precomputing text`

The GPT-OSS text encoder is **~6GB in MXFP4**, but the 24GB preset sets `disable_mxfp4: true`, which **dequantizes it to bf16 (~40GB in system RAM)**. That does not fit on a 16GB GPU as a single model. Latent precompute only uses the VAE; text precompute must run the TE once per image to build disk caches (same idea as Ostris).

With `disable_mxfp4: true`, text precompute runs on **CPU** by default (slow once, then training ignores the TE). On Blackwell (RTX 50xx) you can usually keep MXFP4 and speed up precompute:

```bash
--set model.disable_mxfp4=false
```

Requires `pip install kernels>=0.12.0 triton>=3.4.0`. Training still reads cached embeddings — the TE is not used in the training loop.

**Note:** `kernels>=0.15` is incompatible with `transformers` 5.9 (import crash). Use `kernels>=0.12.0,<0.15`, or uninstall `kernels` if you use `disable_mxfp4: true` (CPU text precompute).

### `ValueError: Either a revision or a version must be specified`

Your ComfyUI venv likely has `kernels` 0.15+ from an MXFP4 install attempt. Fix:

```bash
pip install 'kernels>=0.12.0,<0.15'
# or, if you are not using MXFP4:
pip uninstall -y kernels kernels-data
```

Then re-run `train.py`.

## License

**LensTrainer-LoboForge** (this repository’s code) is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, modify, and share it for **noncommercial** purposes at no charge.

**Commercial use** (paid services, resale, bundling in commercial products, etc.) requires a separate license — see [COMMERCIAL.md](COMMERCIAL.md).

Microsoft Lens, GPT-OSS, and other **model weights** are not covered by this license; they remain under their respective terms on Hugging Face and other upstream sources.
