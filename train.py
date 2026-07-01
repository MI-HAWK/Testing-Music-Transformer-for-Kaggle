import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from pathlib import Path

from utils import load_config, validate_config, set_seed, get_lr, get_param_groups, count_parameters, AverageMeter, CheckpointManager, WandBLogger
from model import CPTransformer
from dataloader import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)
    set_seed(cfg["seed"])
    
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            default_backend = "gloo" if os.name == "nt" else "nccl"
        else:
            device = torch.device("cpu")
            default_backend = "gloo"
            
        backend = cfg.get("backend", default_backend)
        try:
            dist.init_process_group(backend=backend, device_id=device)
        except TypeError:
            dist.init_process_group(backend=backend) # fallback for older pytorch
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device_type = "cuda" if device.type == "cuda" else "cpu"
        
        
    if is_distributed:
        dist.barrier()
        
    model = CPTransformer(cfg).to(device)
    if rank == 0:
        count_parameters(model)
        
    # if not args.sanity:
    #     try:
    #         model = torch.compile(model, mode="default")
    #     except Exception as e:
    #         if rank == 0: print(f"torch.compile failed: {e}")
            
    if is_distributed:
        # find_unused_parameters=False in DDP wrapper
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)
        
    param_groups = get_param_groups(model.module if is_distributed else model, cfg["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler("cuda", enabled=device_type=="cuda")
    
    train_loader = get_dataloader(cfg, split="train", shuffle=True)
    val_loader = get_dataloader(cfg, split="validation", shuffle=False)
    
    if args.sanity:
        cfg["epochs"] = 1
        cfg["val_every_steps"] = 10
        print("Sanity check mode enabled: taking 10 steps.")
        
    logger = WandBLogger(enabled=args.wandb and rank == 0)
    if args.wandb and rank == 0:
        import wandb
        wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], config=cfg, resume="allow")
        
    chkpt_mgr = CheckpointManager(cfg["checkpoint_dir"], cfg["keep_last_n_checkpoints"])
    
    step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    
    if args.resume:
        if rank == 0: print(f"Resuming from {args.resume}")
        chkpt = torch.load(args.resume, map_location=device)
        (model.module if is_distributed else model).load_state_dict(chkpt["model_state_dict"])
        optimizer.load_state_dict(chkpt["optimizer_state_dict"])
        scaler.load_state_dict(chkpt["scaler_state_dict"])
        step = chkpt["step"]
        start_epoch = chkpt["epoch"]
        best_val_loss = chkpt.get("best_val_loss", float("inf"))
        patience_counter = chkpt.get("patience_counter", 0)
        
    total_steps = cfg["epochs"] * (len(train_loader) // cfg["grad_accum_steps"])
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    
    optimizer.zero_grad()
    
    for epoch in range(start_epoch, cfg["epochs"]):
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
            
        model.train()
        loss_meter = AverageMeter()
        
        for batch_idx, batch in enumerate(train_loader):
            input_seq = batch["input"].to(device, non_blocking=True)
            target_seq = batch["target"].to(device, non_blocking=True)
            
            lr = get_lr(step, total_steps, warmup_steps, cfg["lr"], cfg["lr_min"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                
            with autocast(device_type, dtype=torch.float16 if device_type == "cuda" else torch.bfloat16, enabled=device_type=="cuda"):
                loss_dict = model(input_seq, target_seq)
                loss = loss_dict["total"] / cfg["grad_accum_steps"]
                
            scaler.scale(loss).backward()
            loss_meter.update(loss.item() * cfg["grad_accum_steps"])
            
            if (batch_idx + 1) % cfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                
                if rank == 0 and step % 50 == 0:
                    scale_val = scaler.get_scale()
                    metrics = {
                        "train_loss": loss_meter.avg,
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                        "scaler_scale": scale_val,
                        "family_loss": loss_dict["family"].item(),
                        "tempo_loss": loss_dict["tempo"].item(),
                        "position_loss": loss_dict["position_bar"].item(),
                        "pitch_loss": loss_dict["pitch"].item(),
                        "duration_loss": loss_dict["duration"].item(),
                        "velocity_loss": loss_dict["velocity"].item(),
                    }
                    print(f"Ep {epoch} Stp {step} | Loss {loss_meter.avg:.4f} | LR {lr:.2e} | GNorm {grad_norm.item():.2f} | Scale {scale_val}")
                    logger.log(metrics, step)
                    
                if args.sanity and step >= 10:
                    print("Sanity check passed.")
                    if is_distributed: dist.destroy_process_group()
                    return
                    

                if step % cfg["val_every_steps"] == 0:
                    model.eval()
                    val_loss_meter = AverageMeter()
                    with torch.no_grad():
                        for val_batch in val_loader:
                            v_input = val_batch["input"].to(device, non_blocking=True)
                            v_target = val_batch["target"].to(device, non_blocking=True)
                            with autocast(device_type, dtype=torch.float16 if device_type == "cuda" else torch.bfloat16, enabled=device_type=="cuda"):
                                v_loss_dict = model(v_input, v_target)
                            val_loss_meter.update(v_loss_dict["total"].item())
                            
                    v_loss = val_loss_meter.avg
                    if is_distributed:
                        v_loss_t = torch.tensor(v_loss).to(device)
                        dist.all_reduce(v_loss_t, op=dist.ReduceOp.SUM)
                        v_loss = (v_loss_t / dist.get_world_size()).item()
                        
                    if rank == 0:
                        print(f"Validation at step {step}: val_loss={v_loss:.4f}")
                        logger.log({"val_loss": v_loss}, step)
                        
                        # Early stopping and best model
                        if v_loss < best_val_loss - cfg["early_stopping"]["min_delta"]:
                            best_val_loss = v_loss
                            patience_counter = 0
                            best_path = Path(cfg["checkpoint_dir"]) / "best_model.pt"
                            torch.save({
                                "step": step,
                                "epoch": epoch,
                                "model_state_dict": (model.module if is_distributed else model).state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scaler_state_dict": scaler.state_dict(),
                                "config": cfg,
                                "val_loss": v_loss,
                                "best_val_loss": best_val_loss,
                                "patience_counter": patience_counter
                            }, best_path)
                            print(f"New best model at step {step}, val_loss={v_loss:.4f}")
                        else:
                            patience_counter += 1
                    # Sync early stopping decision and best val loss across all GPUs
                    if is_distributed:
                        p_tensor = torch.tensor(patience_counter, dtype=torch.long, device=device)
                        b_tensor = torch.tensor(best_val_loss, dtype=torch.float32, device=device)
                        dist.broadcast(p_tensor, src=0)
                        dist.broadcast(b_tensor, src=0)
                        patience_counter = p_tensor.item()
                        best_val_loss = b_tensor.item()
                        
                    if patience_counter >= cfg["early_stopping"]["patience"]:
                        if rank == 0: print(f"Early stopping at step {step}")
                        if args.wandb and rank == 0: logger.finish()
                        if is_distributed: dist.destroy_process_group()
                        return
                                
                    model.train()
                    
                if step % cfg["save_every_steps"] == 0 and rank == 0:
                    chkpt_path = Path(cfg["checkpoint_dir"]) / f"step_{step}.pt"
                    torch.save({
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": (model.module if is_distributed else model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "config": cfg,
                        "val_loss": v_loss if 'v_loss' in locals() else float('inf'),
                        "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter
                    }, chkpt_path)
                    chkpt_mgr.track_checkpoint(chkpt_path)
                    
    logger.finish()
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
