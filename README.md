# CP Music Transformer

End-to-end Compound Word (CP) Transformer for Pop piano music generation in PyTorch. 
Based on "Compound Word Transformer: Learning to Compose Full-Song Music over Dynamic Directed Hypergraphs" (Hsiao et al., 2021).
Adapted for the MAESTRO dataset (solo piano, no chord labels).

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

**Step 4 — Training with WandB:**
```bash
wandb login
torchrun --nproc_per_node=1 train.py --config config.yaml --wandb
```

**Step 5 — Resume:**
```bash
torchrun --nproc_per_node=1 train.py --config config.yaml --resume checkpoints/step_5000.pt --wandb
```

**Step 6 — Generate:**
```bash
python generate.py --checkpoint checkpoints/best_model.pt --max_len 1024 --output generated.mid
# Temperature sweep:
python generate.py --checkpoint checkpoints/best_model.pt --tau_scale 0.8 --output gen_conservative.mid
python generate.py --checkpoint checkpoints/best_model.pt --tau_scale 1.3 --output gen_creative.mid
```

**Step 7 — Convert to audio:**
```bash
python midi_to_audio.py --input generated.mid --output generated.wav
python midi_to_audio.py --input generated.mid --output generated.mp3 --mp3
```

**Step 8 — Evaluate:**
```bash
# Single reference:
python evals.py --generated generated.mid --reference maestro-v3.0.0/2004/ref.midi

# Population-level FMD (generate 50 first):
for i in $(seq 1 50); do
  python generate.py --checkpoint checkpoints/best_model.pt --output generated_dir/gen_$i.mid --seed $i
done
python evals.py --generated_dir ./generated_dir/ --dataset_dir ./maestro-v3.0.0/ \
  --split test --checkpoint checkpoints/best_model.pt --fmd --self_sim
```

## Kaggle setup
```python
!cp -r /kaggle/input/cp-music-transformer /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -r requirements.txt
# tokenizer.py auto-downloads MAESTRO if not present
!torchrun --nproc_per_node=2 train.py --config config.yaml --wandb
```
Settings → Accelerator → GPU T4 x2. Checkpoints save to `/kaggle/working/`.

## Config quick-reference
| Key | Default | Change for... |
|-----|---------|---------------|
| n_layers | 12 | Set 6 for faster iteration |
| attention_type | linear | Set "standard" to debug |
| pos_encoding | rope | Set "sinusoidal" for ablation |
| weight_tying | true | Set false to ablate |
| batch_size | 16 | Reduce to 8 if OOM; increase grad_accum_steps to 4 |
| early_stopping.patience | 10 | Set 3 for fast runs |
| chunk_stride | 256 | Set 512 for non-overlapping |
| label_smoothing | 0.1 | Set 0 to disable |
| tau_scale (generate.py) | 1.0 | <1 conservative, >1 creative |
