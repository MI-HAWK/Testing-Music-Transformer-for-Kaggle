import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from model import CPTransformer
from utils import set_seed
import miditok
import numpy as np

def top_p_sampling(logits, top_p, temperature=1.0):
    logits = logits / temperature
    if top_p >= 1.0:
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--output", type=str, default="generated.mid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau_scale", type=float, default=1.0)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    chkpt = torch.load(args.checkpoint, map_location=device)
    cfg = chkpt["config"]
    
    model = CPTransformer(cfg).to(device)
    
    state_dict = chkpt["model_state_dict"]
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()
    
    # Load tokenizer
    cache_path = Path("cache")
    tokenizer_path = cache_path / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run tokenizer.py first.")
        
    print(f"Loading tokenizer from {cache_path / 'tokenizer.json'}")
    tokenizer = miditok.CPWord(params=str(cache_path / "tokenizer.json"))
    
    ign = cfg["special_tokens"]["ignore_idx"]
    pad = cfg["special_tokens"]["pad_idx"]
    eos_fam = cfg["special_tokens"]["eos_family_idx"]
    bos_fam = cfg["special_tokens"]["bos_family_idx"]
    note_fam = cfg["special_tokens"]["note_family_idx"]
    metric_fam = cfg["special_tokens"]["metric_family_idx"]
    
    active_slots_map = {
        note_fam: [2, 3, 4, 5], # position_bar, pitch, duration, velocity
        metric_fam: [1]         # tempo
    }
    
    # Initialize sequence with BOS compound word
    seq = [[bos_fam, ign, ign, ign, ign, ign]]
    input_seq = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0) # [1, 1, 6]
    
    print(f"Generating sequence of max length {args.max_len}...")
    
    cache = None
    with torch.no_grad():
        for t in range(args.max_len):
            if cache is None:
                cur_seq = input_seq
                start_pos = 0
            else:
                cur_seq = input_seq[:, -1:]
                start_pos = input_seq.shape[1] - 1
                # Clamp so abs_pos_emb only indexes trained slots [0, max_seq_len-1].
                # The KV-cache still carries the full history; only the positional
                # signal degrades gracefully instead of hitting random embeddings.
                max_trained_pos = cfg["max_seq_len"] - 1
                if start_pos > max_trained_pos:
                    if start_pos == max_trained_pos + 1:  # warn once
                        print(f"Warning: generation exceeded trained positional range "
                              f"({cfg['max_seq_len']} tokens). Clamping pos embedding "
                              f"to slot {max_trained_pos}.")
                    start_pos = max_trained_pos
                
            # Forward pass
            B, T_cur, _ = cur_seq.shape
            
            e_f_all = model.emb_family(cur_seq[:, :, 0])
            e_t_all = model.emb_tempo(cur_seq[:, :, 1])
            e_p_all = model.emb_pos(cur_seq[:, :, 2])
            e_pi_all = model.emb_pitch(cur_seq[:, :, 3])
            e_d_all = model.emb_dur(cur_seq[:, :, 4])
            e_v_all = model.emb_vel(cur_seq[:, :, 5])
            
            x = torch.cat([e_f_all, e_t_all, e_p_all, e_pi_all, e_d_all, e_v_all], dim=-1)
            h = model.w_in(x)
            
            t_pos = torch.arange(start_pos, start_pos + T_cur, device=h.device)
            h = h + model.abs_pos_emb(t_pos)
                
            new_caches = []
            for i, layer in enumerate(model.layers):
                layer_cache = cache[i] if cache is not None else None
                h, new_c = layer(h, layer_cache)
                new_caches.append(new_c)
            cache = new_caches
                
            h = model.norm(h)
            
            h_t = h[:, -1, :] # [1, d_model]
            
            # Stage 1 - sample family using h_t
            logits_f_last = model.w_f(h_t) # [1, vs_family]
            
            l_f = logits_f_last.squeeze(0).clone()
            
            # BOS/PAD/IGNORE logits zeroed for family
            l_f[ign] = -float('Inf')
            l_f[pad] = -float('Inf')
            l_f[bos_fam] = -float('Inf')
            
            tau_f = cfg["sampling"]["family"]["tau"] * args.tau_scale
            top_p_f = cfg["sampling"]["family"]["top_p"]
            
            f_hat = top_p_sampling(l_f.unsqueeze(0), top_p=top_p_f, temperature=tau_f).item()
            
            if f_hat == eos_fam:
                print(f"EOS generated at step {t+1} of max {args.max_len}.")
                break
                
            new_cp = [f_hat, ign, ign, ign, ign, ign]
            
            # Stage 2 - condition on f_hat and h_t
            e_f_sampled = model.emb_family(torch.tensor([f_hat], device=device)) # [1, es_family]
            h_cond = torch.cat([h_t, e_f_sampled], dim=-1) # [1, d_model + es_family]
            
            # Now compute stage 2 logits for this specific step!
            l_t = model.w_out_tempo(h_cond)
            l_p = model.w_out_pos(h_cond)
            l_pi = model.w_out_pitch(h_cond)
            l_d = model.w_out_dur(h_cond)
            l_v = model.w_out_vel(h_cond)
            
            logits_list = [None, l_t, l_p, l_pi, l_d, l_v]
            keys = ["family", "tempo", "position_bar", "pitch", "duration", "velocity"]
            
            active = active_slots_map.get(f_hat, [])
            for slot_idx in active:
                l_slot = logits_list[slot_idx][0].clone()
                # DO NOT mask out 0 for tempo or position_bar when generating a metric token! 
                # 0 means Ignore_None for Tempo, and Bar_None for Position. Both are used in Bar tokens!
                if slot_idx in [1, 2] and f_hat == metric_fam:
                    pass
                else:
                    l_slot[ign] = -float('Inf')
                l_slot[pad] = -float('Inf')
                
                key = keys[slot_idx]
                tau = cfg["sampling"][key]["tau"] * args.tau_scale
                top_p = cfg["sampling"][key]["top_p"]
                
                sampled_val = top_p_sampling(l_slot.unsqueeze(0), top_p=top_p, temperature=tau).item()
                if key == "position_bar":
                    probs = torch.nn.functional.softmax(l_slot, dim=-1)
                    top_probs, top_indices = torch.topk(probs, 5)
                    print(f"Position Top 5 Probs: {top_probs.tolist()} at indices {top_indices.tolist()}")
                new_cp[slot_idx] = sampled_val
                
            new_cp_tensor = torch.tensor([new_cp], dtype=torch.long, device=device).unsqueeze(0)
            input_seq = torch.cat([input_seq, new_cp_tensor], dim=1)
            
        else:
            print(f"Warning: max_len={args.max_len} reached without EOS — piece may be truncated.")
            
    # Remove BOS
    gen_seq = input_seq[0, 1:].cpu().numpy() # [T, 6]
    
    # We need to map our token indices back to miditok's internal token representation.
    # Our indices: shifted by 2 from miditok (so miditok_idx = our_idx + 2 for duration? No, for all except family)
    # Wait, in tokenizer.py:
    # Family: "Note"=note_family_idx, "Metric"=metric_family_idx
    # Other features: we mapped them to index + 2 or argmin + 2.
    # Miditok CPWord tokens: Family, Position, Pitch, Velocity, Duration, Tempo
    # miditok indices:
    # tok_id = mapped - 2 (for position, pitch, duration, velocity, tempo)
    
    # To use tokenizer.tokens_to_midi, we must build a list of miditok event objects or strings
    # Or just construct the (T, 6) tensor that miditok natively uses!
    # Let's map our (T, 6) tensor back to miditok's (T, 6) tensor
    
    # Miditok expected order: 0:Family, 1:Position, 2:Pitch, 3:Velocity, 4:Duration, 5:Tempo
    # Our order: 0:family, 1:tempo, 2:position, 3:pitch, 4:duration, 5:velocity
    
    miditok_tokens = np.zeros((gen_seq.shape[0], 6), dtype=np.int32)
    
    for t in range(gen_seq.shape[0]):
        # Family
        f_val = gen_seq[t, 0]
        # miditok special tokens are 0-3 (PAD, BOS, EOS, MASK). 
        # Actually, if we just give it miditok's exact IDs, it works perfectly.
        # But wait! We mapped string to our IDs directly. We need to map our IDs back to string or miditok IDs.
        # In tokenizer.py:
        # Note -> cfg["special_tokens"]["note_family_idx"] (which is 4)
        # Metric -> cfg["special_tokens"]["metric_family_idx"] (which is 5)
        # In miditok's Family vocab, Note might be 4 and Metric might be 5! 
        # But to be safe, let's reverse lookup miditok's vocab!
        
        # We can just reverse the exact logic:
        pass
        
    # Let's just do it cleanly:
    inv_vocabs = [{v: k for k, v in vocab.items()} for vocab in tokenizer.vocab]
    
    # Build list of token IDs in miditok format    # Re-convert to miditok tokens
    miditok_seq = []
    
    def safe_lookup(vocab_idx, prefix, val):
        key = f"{prefix}_{val}"
        if key in tokenizer.vocab[vocab_idx]:
            return tokenizer.vocab[vocab_idx][key]
        return list(tokenizer.vocab[vocab_idx].values())[0]

    for step in range(len(gen_seq)):
        cp = gen_seq[step]
        f_val = cp[0]
        f_str = ""
        if f_val == note_fam: f_str = "Family_Note"
        elif f_val == metric_fam: f_str = "Family_Metric"
        else: continue
            
        miditok_cp = [0]*6
        miditok_cp[0] = tokenizer.vocab[0][f_str]
        
        if f_val == note_fam:
            miditok_cp[1] = tokenizer.vocab[1].get("Ignore_None", 0)
            
            pitch_val = cp[3] + 21 - 2
            miditok_cp[2] = safe_lookup(2, "Pitch", pitch_val)
            
            vel_idx = cp[5] - 2
            vel_val = tokenizer.velocities[vel_idx] if vel_idx >= 0 and vel_idx < len(tokenizer.velocities) else 0
            miditok_cp[3] = safe_lookup(3, "Velocity", vel_val)
            
            dur_id = cp[4] + 2 if cp[4] >= 2 else list(tokenizer.vocab[4].values())[0]
            dur_id = min(dur_id, len(tokenizer.vocab[4]) - 1)
            miditok_cp[4] = dur_id
            miditok_cp[5] = tokenizer.vocab[5].get("Ignore_None", 0)
            
        elif f_val == metric_fam:
            if cp[2] == 0:
                miditok_cp[1] = safe_lookup(1, "Bar", "None")
            else:
                pos_val = cp[2] - 2
                miditok_cp[1] = safe_lookup(1, "Position", pos_val)
            
            miditok_cp[2] = tokenizer.vocab[2].get("Ignore_None", 0)
            miditok_cp[3] = tokenizer.vocab[3].get("Ignore_None", 0)
            miditok_cp[4] = tokenizer.vocab[4].get("Ignore_None", 0)
            
            tempo_idx = cp[1] - 2
            if tempo_idx >= 0 and tempo_idx < len(tokenizer.tempos):
                tempo_val = tokenizer.tempos[tempo_idx]
                miditok_cp[5] = safe_lookup(5, "Tempo", tempo_val)
            else:
                miditok_cp[5] = tokenizer.vocab[5].get("Ignore_None", 0)
            
        miditok_seq.append(miditok_cp)
        
    # Convert to midi
    from miditok.classes import TokSequence
    ts = miditok_seq
    try:
        midi = tokenizer.tokens_to_midi([TokSequence(ids=ts, ids_bpe_encoded=False)])
    except Exception as e:
        print("Failed to convert to midi. Exception:", e)
        raise e
                    
    midi.dump(args.output)
    print(f"Saved generated MIDI to {args.output}")

if __name__ == "__main__":
    main()
