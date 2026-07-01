import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from pathlib import Path

class CPDataset(Dataset):
    def __init__(self, cache_dir: str, split: str, max_seq_len: int, chunk_stride: int, special_tokens: dict):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.chunk_size = max_seq_len + 1
        self.chunk_stride = chunk_stride
        self.special_tokens = special_tokens
        
        # Pad compound word
        ign = self.special_tokens["ignore_idx"]
        pad = self.special_tokens["pad_idx"]
        self.pad_compound_word = torch.tensor([pad, pad, pad, pad, pad, pad], dtype=torch.long)
        
        self.chunks = []
        cache_path = Path(cache_dir)
        
        from tokenizer import get_split_paths
        paths = get_split_paths(split)
        
        for p in paths:
            if p.exists():
                file_tensor = torch.load(p, weights_only=True) # [L, 6]
                L = file_tensor.shape[0]
                if L < 2:
                    continue # Should have at least BOS and EOS
                    
                start_indices = list(range(0, max(1, L - 1), self.chunk_stride))
                for idx, i in enumerate(start_indices):
                    is_last = (idx == len(start_indices) - 1)
                    self.chunks.append((p, i, is_last))
                    
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        path, start_i, is_last = self.chunks[idx]
        file_tensor = torch.load(path, weights_only=True)
        L = file_tensor.shape[0]
        
        chunk = file_tensor[start_i : start_i + self.chunk_size].clone()
        
        if not is_last:
            # Replace EOS token if it falls within this non-final chunk
            eos_idx_in_chunk = L - 1 - start_i
            if 0 <= eos_idx_in_chunk < len(chunk):
                chunk[eos_idx_in_chunk] = self.pad_compound_word
                
        actual_len = len(chunk)
        if actual_len < self.chunk_size:
            pad_len = self.chunk_size - actual_len
            pads = self.pad_compound_word.unsqueeze(0).expand(pad_len, 6)
            chunk = torch.cat([chunk, pads], dim=0)
            
        input_seq = chunk[:-1]
        target_seq = chunk[1:]
        
        mask = torch.zeros(self.max_seq_len, dtype=torch.float32)
        mask[:actual_len - 1] = 1.0
        
        return {
            "input": input_seq,
            "target": target_seq,
            "mask": mask
        }

def get_dataloader(cfg, split: str, shuffle: bool = True):
    dataset = CPDataset(
        cache_dir=cfg["midi_cache_dir"],
        split=split,
        max_seq_len=cfg["max_seq_len"],
        chunk_stride=cfg["chunk_stride"],
        special_tokens=cfg["special_tokens"]
    )
    
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
        
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=(split == "train")
    )
    return loader
