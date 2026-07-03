import argparse
import numpy as np
import pretty_midi
import scipy.stats
import scipy.linalg
from pathlib import Path
import torch
from tabulate import tabulate
import matplotlib.pyplot as plt
import csv

def compute_basic_metrics(midi_path: str):
    pm = pretty_midi.PrettyMIDI(midi_path)
    
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
            
    if len(notes) == 0:
        return None
        
    notes.sort(key=lambda x: x.start)
    
    # Pitch Class Histogram Entropy
    pitch_classes = [n.pitch % 12 for n in notes]
    counts = np.bincount(pitch_classes, minlength=12)
    probs = counts / np.sum(counts)
    pc_entropy = scipy.stats.entropy(probs)
    
    # Pitch Range
    pitches = [n.pitch for n in notes]
    pitch_range = max(pitches) - min(pitches)
    
    # IOI (Inter-Onset Interval)
    onsets = np.array([n.start for n in notes])
    iois = np.diff(onsets)
    iois = iois[iois > 0] # Filter out chords
    avg_ioi = np.mean(iois) if len(iois) > 0 else 0
    groove_consistency = np.std(iois) if len(iois) > 0 else 0
    
    # Velocity Entropy
    velocities = [n.velocity for n in notes]
    vel_bins = np.histogram(velocities, bins=32, range=(0, 127))[0]
    vel_probs = vel_bins / np.sum(vel_bins)
    vel_entropy = scipy.stats.entropy(vel_probs)
    
    # Note Density (approximate assuming 4/4 and 120BPM if no tempo)
    length_sec = pm.get_end_time()
    bars = length_sec / (4 * 60 / 120.0) # rough estimate
    note_density = len(notes) / max(1, bars)
    
    return {
        "pc_entropy": pc_entropy,
        "pitch_range": pitch_range,
        "avg_ioi": avg_ioi,
        "groove_consistency": groove_consistency,
        "vel_entropy": vel_entropy,
        "note_density": note_density
    }

def get_chroma_features(pm):
    # Bar-level chroma
    return pm.get_chroma().T # approx [time, 12]

def compute_fmd(generated_h, ref_h):
    mu_gen = np.mean(generated_h, axis=0)
    sigma_gen = np.cov(generated_h, rowvar=False)
    
    mu_ref = np.mean(ref_h, axis=0)
    sigma_ref = np.cov(ref_h, rowvar=False)
    
    diff = mu_gen - mu_ref
    
    covmean, _ = scipy.linalg.sqrtm(sigma_gen.dot(sigma_ref), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fmd = diff.dot(diff) + np.trace(sigma_gen + sigma_ref - 2 * covmean)
    return fmd

def extract_features(model, midi_path, cfg):
    """Run a CP sequence through the model and return the mean hidden state.
    Uses the same forward pass as CPTransformer.forward() with absolute positional
    embeddings, clamped to the trained max_seq_len range."""
    from tokenizer import CPTokenizer
    tok = CPTokenizer(cfg)
    seq = tok.process_midi(Path(midi_path))
    if seq is None: return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq = seq.unsqueeze(0).to(device)  # [1, L, 6]

    with torch.no_grad():
        vs = cfg["vocab_sizes"]
        # Clamp token indices to stay within each vocab, then truncate to trained
        # positional range so abs_pos_emb is never called with untrained slots.
        max_pos = cfg["max_seq_len"]
        f_in  = torch.clamp(seq[:, :max_pos, 0], max=vs["family"] - 1)
        t_in  = torch.clamp(seq[:, :max_pos, 1], max=vs["tempo"] - 1)
        p_in  = torch.clamp(seq[:, :max_pos, 2], max=vs["position_bar"] - 1)
        pi_in = torch.clamp(seq[:, :max_pos, 3], max=vs["pitch"] - 1)
        d_in  = torch.clamp(seq[:, :max_pos, 4], max=vs["duration"] - 1)
        v_in  = torch.clamp(seq[:, :max_pos, 5], max=vs["velocity"] - 1)

        e_f  = model.emb_family(f_in)
        e_t  = model.emb_tempo(t_in)
        e_p  = model.emb_pos(p_in)
        e_pi = model.emb_pitch(pi_in)
        e_d  = model.emb_dur(d_in)
        e_v  = model.emb_vel(v_in)

        x = torch.cat([e_f, e_t, e_p, e_pi, e_d, e_v], dim=-1)
        h = model.w_in(x)

        T = h.shape[1]
        t_pos = torch.arange(T, device=h.device)  # [0 .. T-1], always within trained range
        h = h + model.abs_pos_emb(t_pos)

        for layer in model.layers:
            h, _ = layer(h, cache=None)  # CPLayer.forward(x, cache=None)
        h = model.norm(h)

    return h[0].cpu().numpy().mean(axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str)
    parser.add_argument("--reference", type=str)
    parser.add_argument("--generated_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--fmd", action="store_true")
    parser.add_argument("--self_sim", action="store_true")
    args = parser.parse_args()
    
    if args.generated and args.reference:
        gen_metrics = compute_basic_metrics(args.generated)
        ref_metrics = compute_basic_metrics(args.reference)
        
        table = []
        for k in gen_metrics.keys():
            table.append([k, f"{gen_metrics[k]:.4f}", f"{ref_metrics[k]:.4f}"])
        print(tabulate(table, headers=["Metric", "Generated", "Reference"]))
        
    elif args.generated_dir and args.dataset_dir:
        gen_dir = Path(args.generated_dir)
        gen_files = list(gen_dir.glob("*.mid"))
        
        if len(gen_files) < 50:
            print(f"Warning: Only {len(gen_files)} files found. 50+ recommended for FMD.")
            
        # load config and model
        import yaml
        from utils import load_config
        
        if args.fmd:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            chkpt = torch.load(args.checkpoint, map_location=device)
            cfg = chkpt["config"]
            from model import CPTransformer
            model = CPTransformer(cfg).to(device)
            model.load_state_dict(chkpt["model_state_dict"])
            model.eval()
            
            gen_h = []
            for f in gen_files:
                h = extract_features(model, str(f), cfg)
                if h is not None: gen_h.append(h)
                
            ref_h = []
            csv_path = Path(args.dataset_dir) / "maestro-v3.0.0.csv"
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["split"] == args.split:
                        path = Path(args.dataset_dir) / row["midi_filename"]
                        h = extract_features(model, str(path), cfg)
                        if h is not None:
                            ref_h.append(h)
                            if len(ref_h) >= len(gen_h): break # match sizes for fair comparison
                            
            fmd_score = compute_fmd(np.array(gen_h), np.array(ref_h))
            print(f"Fréchet Music Distance (FMD): {fmd_score:.4f}")
            
    if args.self_sim and args.generated:
        pm = pretty_midi.PrettyMIDI(args.generated)
        chroma = get_chroma_features(pm)
        # downsample to bar level (approximate)
        bar_len = 4 * 100 # 4 beats * 100 frames per beat roughly
        bars = chroma.shape[0] // bar_len
        if bars > 0:
            chroma_bars = np.array([chroma[i*bar_len:(i+1)*bar_len].mean(axis=0) for i in range(bars)])
            sim_matrix = np.dot(chroma_bars, chroma_bars.T)
            norms = np.linalg.norm(chroma_bars, axis=1)
            sim_matrix = sim_matrix / np.outer(norms, norms)
            
            plt.imshow(sim_matrix, cmap='hot', interpolation='nearest')
            plt.title("Self-Similarity Matrix")
            plt.savefig("self_sim.png")
            print("Saved Self-Similarity Matrix to self_sim.png")

if __name__ == "__main__":
    main()
