# CP Music Transformer

An end-to-end generative AI model for expressive piano music, built in PyTorch. 

## About the Project
This project trains an autoregressive Transformer to compose and generate human-like piano performances. Rather than predicting raw audio waves, it learns to generate MIDI events (pitches, velocities, durations, and tempo changes) using a highly efficient tokenization strategy. 

## Inspiration & Paper References
This repository is heavily inspired by the paper **"Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs"** (Hsiao et al., 2021). 

The "Compound Word" (CP) representation groups simultaneous musical attributes (like the pitch, duration, and velocity of a single note) into a single "compound word." This reduces the sequence length by up to 5x compared to standard MIDI tokenization, allowing the Transformer to generate longer, more coherent musical phrases much faster.

### Differences from the Original Paper
While inspired by the CP Transformer, this implementation diverges in several key ways:
1. **Dataset:** The original paper trained on a proprietary dataset of Pop piano arrangements. This repository is adapted for the open-source **MAESTRO v3.0.0** dataset, containing over 200 hours of virtuosic classical piano performances.
2. **Tokenization Structure:** Because MAESTRO does not contain chord labels, we omit the `[chord]` token. Our CP representation consists of 6 parallel tokens per step: `[Family, Tempo, Position, Pitch, Duration, Velocity]`.
3. **Task:** The original paper featured conditional generation (lead sheet to full arrangement). This repo focuses strictly on unconditional generation (composing from scratch).
4. **Architecture:** We employ a standard causal Transformer in PyTorch utilizing `scaled_dot_attention` instead of the Linear Transformer backend with linear attention mentioned in the paper. We also completely decoupled tempo modeling during training to prevent timing collapses.
5. **Distributed Training:** The training loop is fully optimized for multi-GPU setups using PyTorch's Distributed Data Parallel (DDP) framework, allowing for scalable data parallel processing across multiple devices.

## Dataset
This project uses the [MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) v3.0.0 dataset](https://magenta.tensorflow.org/datasets/maestro). It contains over 200 hours of paired audio and MIDI recordings from ten years of International Piano-e-Competition events. The `tokenizer.py` script automatically downloads and processes this dataset for you.

## Setup
```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# On Linux / Kaggle:
sudo apt-get install fluidsynth ffmpeg
# On Windows (optional, only needed for mp3 conversion):
# winget install Gyan.FFmpeg
```

## End-to-end execution

**Step 1 — Tokenize MAESTRO (auto-downloads ~57MB if not present; run once):**
```bash
python tokenizer.py --config config.yaml
```

**Step 2 — Sanity check:**
```bash
torchrun --nproc_per_node=1 train.py --config config.yaml --sanity
```
> **Note for Windows users:** If you encounter a `RuntimeError: use_libuv was requested but PyTorch was build without libuv support` when running `torchrun`, you need to disable libuv. In PowerShell, run `$env:USE_LIBUV="0"` before your command, or in Command Prompt run `set USE_LIBUV=0`.

**Step 3 — Full training:**
```bash
torchrun --nproc_per_node=1 train.py --config config.yaml
```

**Step 4 — Resume:**
```bash
torchrun --nproc_per_node=1 train.py --config config.yaml --resume checkpoints/step_5000.pt
```

**Step 5 — Generate:**
```bash
python generate.py --checkpoint checkpoints/best_model.pt --max_len 1024 --output generated.mid
# Temperature sweep:
python generate.py --checkpoint checkpoints/best_model.pt --tau_scale 0.8 --output gen_conservative.mid
```

## Kaggle setup
```python
!git clone https://github.com/MI-HAWK/Testing-Music-Transformer-for-Kaggle.git
%cd /kaggle/working/repo
!pip install miditok pretty_midi pyyaml
# tokenizer.py auto-downloads MAESTRO if not present
!torchrun --nproc_per_node=2 train.py --config config.yaml
```
Settings → Accelerator → GPU T4 x2. Checkpoints save to `/kaggle/working/`.

## Config quick-reference
| Key | Default | Change for... |
|-----|---------|---------------|
| n_layers | 12 | Set 6 for faster iteration |
| batch_size | 16 | Reduce to 8 if OOM; increase grad_accum_steps to 4 |
| early_stopping.patience | 10 | Set 3 for fast runs |
| chunk_stride | 256 | Set 512 for non-overlapping |
| label_smoothing | 0.1 | Set 0 to disable |
| tau_scale (generate.py) | 1.0 | <1 conservative, >1 creative |
