import math
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from collections import deque
import wandb

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def validate_config(cfg: dict) -> None:
    # vocab_sizes: all 6 types present, all values > 2
    required_types = ["family", "tempo", "position_bar", "pitch", "duration", "velocity"]
    for t in required_types:
        if t not in cfg["vocab_sizes"]:
            raise ValueError(f"Missing vocab size for {t}")
        if cfg["vocab_sizes"][t] <= 2:
            raise ValueError(f"vocab size for {t} must be > 2")
    if cfg["vocab_sizes"]["family"] != 6:
        raise ValueError("vocab_sizes.family must be 6")
        
    for t in required_types:
        if t not in cfg["token_embed_sizes"]:
            raise ValueError(f"Missing token_embed_size for {t}")
            
    st = cfg["special_tokens"]
    if st["ignore_idx"] != 0 or st["pad_idx"] != 1:
        raise ValueError("ignore_idx must be 0, pad_idx must be 1")
        
    required_special = ["eos_family_idx", "bos_family_idx", "note_family_idx", "metric_family_idx"]
    for s in required_special:
        if s not in st:
            raise ValueError(f"Missing special token {s}")
            
    indices = [st["ignore_idx"], st["pad_idx"]] + [st[k] for k in required_special]
    if len(set(indices)) != len(indices):
        raise ValueError("Special token indices must be distinct")
    for idx in indices:
        if not (0 <= idx < cfg["vocab_sizes"]["family"]):
            raise ValueError(f"Special token index {idx} out of range for family vocab")
            
    if not (cfg["tempo_bins"]["min_bpm"] < cfg["tempo_bins"]["max_bpm"]):
        raise ValueError("tempo_bins: min_bpm must be < max_bpm")
    if cfg["tempo_bins"]["n_bins"] <= 0:
        raise ValueError("tempo_bins: n_bins must be > 0")
    if cfg["tempo_bins"]["scale"] not in ["log", "linear"]:
        raise ValueError("tempo_bins: scale must be log or linear")
        
    if cfg["velocity_bins"]["min_val"] != 0 or cfg["velocity_bins"]["max_val"] != 127:
        raise ValueError("velocity_bins: min/max must be 0/127")
    if cfg["velocity_bins"]["n_bins"] <= 0:
        raise ValueError("velocity_bins: n_bins must be > 0")
        
    if len(cfg["duration_bins"]["values"]) != cfg["duration_bins"]["n_bins"]:
        raise ValueError("duration_bins: values length must match n_bins")
        
    if cfg["n_layers"] <= 0 or cfg["n_heads"] <= 0 or cfg["d_model"] % cfg["n_heads"] != 0:
        raise ValueError("Invalid model architecture dimensions")
        
    if cfg["batch_size"] <= 0 or cfg["grad_accum_steps"] <= 0:
        raise ValueError("Batch size and grad_accum_steps must be > 0")
        
    if not (cfg["lr"] > cfg["lr_min"] > 0):
        raise ValueError("Learning rate must satisfy lr > lr_min > 0")
    if not (0 < cfg["warmup_ratio"] < 1):
        raise ValueError("warmup_ratio must be between 0 and 1")
        
    if cfg["keep_last_n_checkpoints"] < 1:
        raise ValueError("keep_last_n_checkpoints must be >= 1")
        
    if cfg["attention_type"] not in ["linear", "standard"]:
        raise ValueError("Invalid attention_type")
    if cfg["pos_encoding"] not in ["rope"]:
        raise ValueError("Invalid pos_encoding, only rope is supported")
        
    if cfg["early_stopping"]["patience"] <= 0 or cfg["early_stopping"]["min_delta"] <= 0:
        raise ValueError("Invalid early stopping params")
        
    if not (0 <= cfg["label_smoothing"] < 1):
        raise ValueError("label_smoothing must be in [0, 1)")

def get_lr(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

def get_param_groups(model: torch.nn.Module, weight_decay: float) -> list:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases, LayerNorm weights, and other 1D parameters should not be decayed
        if param.dim() < 2:
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def count_parameters(model: torch.nn.Module) -> int:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {params / 1e6:.2f}M")
    return params

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / max(1, self.count)

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, keep_n: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self.tracked = deque()
        
    def track_checkpoint(self, path: Path):
        self.tracked.append(path)
        if len(self.tracked) > self.keep_n:
            oldest = self.tracked.popleft()
            if oldest.exists() and oldest.name.startswith("step_"):
                oldest.unlink()

class WandBLogger:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        
    def log(self, metrics: dict, step: int):
        if self.enabled:
            wandb.log(metrics, step=step)
            
    def finish(self):
        if self.enabled:
            wandb.finish()
