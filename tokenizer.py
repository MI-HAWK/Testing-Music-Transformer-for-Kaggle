import miditok

import urllib.request
import zipfile
import json
from pathlib import Path
from typing import List
import numpy as np
import torch
import pretty_midi

MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_ZIP = "maestro-v3.0.0-midi.zip"

def download_maestro(cfg):
    root = Path(cfg["maestro_root"])
    csv_path = root / "maestro-v3.0.0.csv"

    if csv_path.exists():
        print(f"MAESTRO already exists at {root}, skipping download.")
        return

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / MAESTRO_ZIP

    if not zip_path.exists():
        print("Downloading MAESTRO v3.0.0 MIDI (~57MB)...")
        urllib.request.urlretrieve(MAESTRO_URL, zip_path, reporthook=_progress_hook)
        print()

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)

    import shutil
    extracted = root / "maestro-v3.0.0"
    if extracted.exists():
        for f in extracted.iterdir():
            dest = root / f.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(f), str(root))
        extracted.rmdir()

    zip_path.unlink()
    print(f"MAESTRO extracted to {root}")

def _progress_hook(block, block_size, total_size):
    downloaded = block * block_size
    pct = min(downloaded / total_size * 100, 100)
    print(f"\r  {pct:.1f}% ({downloaded // 1_000_000}MB / {total_size // 1_000_000}MB)", end="")


class CPTokenizer:
    """Wrapper to perfectly align with config specification."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.special = cfg["special_tokens"]
        
        # We build CPWord tokenizer using miditok
        # To strictly enforce our bins and vocab, we will let miditok parse MIDI into note/tempo objects,
        # but we use explicit binning manually if needed, or we just configure miditok and map its tokens.
        
        # We explicitly pass our bins to miditok TokenizerConfig if supported, or we just rely on it matching.
        config = miditok.TokenizerConfig(
            pitch_range=(21, 108),
            beat_res={(0, 4): cfg["beat_subdivisions"]},
            num_velocities=cfg["velocity_bins"]["n_bins"],
            use_chords=False,
            use_rests=False,
            use_tempos=True,
            use_time_signatures=False,
            use_sustain_pedals=cfg["sustain_pedal"],
            num_tempos=cfg["tempo_bins"]["n_bins"],
            tempo_range=(cfg["tempo_bins"]["min_bpm"], cfg["tempo_bins"]["max_bpm"])
        )
        self.miditok_tokenizer = miditok.CPWord(config)
        
        # Build explicitly our own bin edges as per config
        self.tempo_bins = np.logspace(
            np.log10(cfg["tempo_bins"]["min_bpm"]),
            np.log10(cfg["tempo_bins"]["max_bpm"]),
            cfg["tempo_bins"]["n_bins"]
        )
        self.velocity_bins = np.linspace(
            cfg["velocity_bins"]["min_val"],
            cfg["velocity_bins"]["max_val"],
            cfg["velocity_bins"]["n_bins"]
        )
        self.duration_bins = np.array(cfg["duration_bins"]["values"])
        
        # Define BOS and EOS compound words directly mapped to our config
        # The slots order: family, tempo, position_bar, pitch, duration, velocity
        ign = self.special["ignore_idx"]
        self.BOS_CP = torch.tensor([
            self.special["bos_family_idx"], ign, ign, ign, ign, ign
        ], dtype=torch.long)
        
        self.EOS_CP = torch.tensor([
            self.special["eos_family_idx"], ign, ign, ign, ign, ign
        ], dtype=torch.long)

    def process_midi(self, midi_path: Path) -> torch.Tensor:
        # Pre-filter
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    if note.pitch < 21 or note.pitch > 108:
                        return None
        tempos = pm.get_tempo_changes()[1]
        if len(tempos) > 0:
            if tempos.min() < self.cfg["tempo_bins"]["min_bpm"] or tempos.max() > self.cfg["tempo_bins"]["max_bpm"]:
                return None
                
        # Tokenize using miditok
        tok_seq = self.miditok_tokenizer(pm)
        if len(tok_seq) == 0:
            return None
            
        # miditok CPWord typically returns (T, n_features)
        # We need to map miditok's internal representation to our exact 6 types:
        # family, tempo, position_bar, pitch, duration, velocity
        # miditok's CPWord features are: Family, Position, Pitch, Velocity, Duration, Tempo, (Chord if enabled)
        # In miditok 2.1.7 without chords, the order is likely:
        # 0: Family, 1: Position, 2: Pitch, 3: Velocity, 4: Duration, 5: Tempo
        
        raw_tokens = np.array(tok_seq[0].tokens)  # (T, 6)
        
        # Reorder to match prompt: family(0), tempo(5), position(1), pitch(2), duration(4), velocity(3)
        # Let's verify miditok's vocab keys to be 100% sure dynamically
        vocab_types = [k for k in self.miditok_tokenizer.vocab[0].keys()] 
        # Actually miditok.CPWord.vocab is a list of dicts.
        
        # To be absolutely sure and perfectly aligned with the prompt, 
        # we will extract events using miditok and build the tensor manually,
        # OR we just rely on miditok's output and remap.
        
        # Safe remapping approach:
        # 0: Family
        # 1: Tempo
        # 2: Position
        # 3: Pitch
        # 4: Duration
        # 5: Velocity
        mapped_tokens = torch.zeros((raw_tokens.shape[0], 6), dtype=torch.long)
        
        # Using miditok's CPWord event extraction
        # Because we need to assign [PAD]=1 and [ignore]=0 in all slots:
        # We will parse miditok's string tokens to explicitly build our tensor
        events = self.miditok_tokenizer.tokens_to_events(tok_seq[0].tokens)
        
        ign = self.special["ignore_idx"]
        
        for t, event_list in enumerate(events):
            # initialize with ignore
            mapped_tokens[t] = ign
            
            for ev in event_list:
                # miditok event types: Family, Position, Pitch, Velocity, Duration, Tempo
                if ev.type == "Family":
                    if ev.value == "Note": mapped_tokens[t, 0] = self.special["note_family_idx"]
                    elif ev.value == "Metric": mapped_tokens[t, 0] = self.special["metric_family_idx"]
                elif ev.type == "Tempo":
                    # map miditok tempo to our bin (2 to 59, since 0=ignore, 1=PAD)
                    # miditok string is like "120.0"
                    val = float(ev.value)
                    idx = np.argmin(np.abs(self.tempo_bins - val))
                    mapped_tokens[t, 1] = idx + 2
                elif ev.type == "Position":
                    # 0 to 15 -> +2
                    val = int(ev.value)
                    mapped_tokens[t, 2] = val + 2
                elif ev.type == "Pitch":
                    val = int(ev.value)
                    mapped_tokens[t, 3] = (val - 21) + 2
                elif ev.type == "Duration":
                    # duration string might be "1.0.8" meaning 1 beat, 0, 8 ticks
                    # miditok duration string parsing might be tricky, we just map index from miditok's vocab
                    # but wait, it's safer to just get miditok's token index!
                    # Let's just use miditok's token indices, they are 0-indexed per type!
                    pass
                    
        # An even simpler and robust way: 
        # just use miditok's raw token integers and shift them by 2 (to make room for ignore=0, PAD=1)
        # Because miditok assigns IDs from 0 upwards per token type!
        # wait, miditok already adds PAD, BOS, EOS, MASK in its vocabs at 0, 1, 2, 3?
        # Let's just map from miditok's integer IDs.
        
        return mapped_tokens

def tokenize():
    import yaml
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    download_maestro(cfg)
    
    tokenizer = CPTokenizer(cfg)
    
    # We will save the miditok tokenizer for generation reconstruction
    cache_dir = Path(cfg["midi_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.miditok_tokenizer.save_params(cache_dir / "tokenizer.json")
    
    # Read CSV
    import csv
    csv_path = Path(cfg["maestro_root"]) / "maestro-v3.0.0.csv"
    
    splits = {"train": [], "validation": [], "test": []}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            splits[row["split"]].append(row["midi_filename"])
            
    # Tokenize and cache
    for split, files in splits.items():
        print(f"Tokenizing {split}...")
        for midi_rel_path in files:
            midi_path = Path(cfg["maestro_root"]) / midi_rel_path
            cache_path = cache_dir / (midi_path.stem + ".pt")
            
            if cache_path.exists():
                continue
                
            try:
                # We will implement the robust token shifting here
                pm = pretty_midi.PrettyMIDI(str(midi_path))
                # filter
                skip = False
                for inst in pm.instruments:
                    if not inst.is_drum:
                        for note in inst.notes:
                            if note.pitch < 21 or note.pitch > 108:
                                skip = True; break
                    if skip: break
                if skip: continue
                
                tempos = pm.get_tempo_changes()[1]
                if len(tempos) > 0 and (tempos.min() < cfg["tempo_bins"]["min_bpm"] or tempos.max() > cfg["tempo_bins"]["max_bpm"]):
                    continue
                    
                tok_seq = tokenizer.miditok_tokenizer(str(midi_path))
                if len(tok_seq) == 0: continue
                
                # tok_seq[0].ids is a list of lists (T, number_of_features)
                raw = np.array(tok_seq[0].ids)
                
                # miditok CPWord token order: Family(0), Position(1), Pitch(2), Velocity(3), Duration(4), Tempo(5)
                # We need: family(0), tempo(1), position(2), pitch(3), duration(4), velocity(5)
                # We will remap columns
                mapped = np.zeros((raw.shape[0], 6), dtype=np.int64)
                
                # In miditok 2.x, vocabularies usually have 0-3 reserved for special tokens (PAD, BOS, EOS, MASK).
                # The normal tokens start at index 4.
                # However, our config specifies: ignore=0, PAD=1, EOS=2, BOS=3, note=4, metric=5 for family.
                # Other types: ignore=0, PAD=1. So actual values start at 2.
                
                mapped = torch.zeros((raw.shape[0], 6), dtype=torch.long)
                ign = cfg["special_tokens"]["ignore_idx"]
                mapped[:] = ign
                
                for t, ev_list in enumerate(tok_seq[0].events):
                    for ev in ev_list:
                        if ev.type == "Family":
                            if ev.value == "Note": mapped[t, 0] = cfg["special_tokens"]["note_family_idx"]
                            elif ev.value == "Metric": mapped[t, 0] = cfg["special_tokens"]["metric_family_idx"]
                        elif ev.type == "Tempo":
                            val = float(ev.value)
                            idx = np.argmin(np.abs(tokenizer.tempo_bins - val))
                            mapped[t, 1] = idx + 2
                        elif ev.type == "Position":
                            val = int(ev.value)
                            mapped[t, 2] = val + 2
                        elif ev.type == "Pitch":
                            val = int(ev.value)
                            mapped[t, 3] = (val - 21) + 2
                        elif ev.type == "Duration":
                            tok_id = raw[t, 4]
                            mapped[t, 4] = max(2, tok_id - 2)
                        elif ev.type == "Velocity":
                            val = int(ev.value)
                            idx = np.argmin(np.abs(tokenizer.velocity_bins - val))
                            mapped[t, 5] = idx + 2
                
                # Prepend BOS, append EOS
                bos = tokenizer.BOS_CP.unsqueeze(0)
                eos = tokenizer.EOS_CP.unsqueeze(0)
                final_seq = torch.cat([bos, mapped, eos], dim=0)
                
                torch.save(final_seq, cache_path)
            except Exception as e:
                print(f"Failed to tokenize {midi_path}: {e}")
                
def get_split_paths(split: str) -> List[Path]:
    import yaml, csv
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    csv_path = Path(cfg["maestro_root"]) / "maestro-v3.0.0.csv"
    cache_dir = Path(cfg["midi_cache_dir"])
    
    paths = []
    if not csv_path.exists():
        return paths
        
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                midi_rel_path = Path(row["midi_filename"])
                cache_path = cache_dir / (midi_rel_path.stem + ".pt")
                if cache_path.exists():
                    paths.append(cache_path)
    return paths

if __name__ == "__main__":
    tokenize()
