import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def causal_linear_attention(q, k, v, cache=None):
    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()

    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0

    # Normalize to prevent cumsum explosion -> NaN
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    EPS = 1e-6

    if cache is not None:
        S, z = cache
        S, z = S.float(), z.float()
        kv = torch.einsum('b h t k, b h t v -> b h k v', k, v)
        S_new = S + kv
        z_new = z + k.squeeze(2)

        num = torch.einsum('b h t k, b h k v -> b h t v', q, S_new)
        den = torch.einsum('b h t k, b h k -> b h t', q, z_new).clamp(min=EPS)
        out = num / den.unsqueeze(-1)
        new_cache = (S_new.to(dtype), z_new.to(dtype))
    else:
        kv = torch.einsum('b h t k, b h t v -> b h t k v', k, v)
        S = torch.cumsum(kv, dim=2)
        z = torch.cumsum(k, dim=2)

        num = torch.einsum('b h t k, b h t k v -> b h t v', q, S)
        den = torch.einsum('b h t k, b h t k -> b h t', q, z).clamp(min=EPS)
        out = num / den.unsqueeze(-1)
        new_cache = (S[:, :, -1, :, :].to(dtype), z[:, :, -1, :].to(dtype))

    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(dtype), new_cache


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        inner_dim = int(d_ff * 2/3 / 64) * 64
        self.w1 = nn.Linear(d_model, inner_dim, bias=False)
        self.w2 = nn.Linear(d_model, inner_dim, bias=False)
        self.w3 = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class CPAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.d_model = cfg["d_model"]
        self.head_dim = self.d_model // self.n_heads

        self.wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wv = nn.Linear(self.d_model, self.d_model, bias=False)
        self.wo = nn.Linear(self.d_model, self.d_model, bias=False)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out, new_cache = causal_linear_attention(q, k, v, cache)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.drop(self.wo(out)), new_cache


class CPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CPAttention(cfg)
        self.ffn = SwiGLU(cfg["d_model"], cfg["d_ff"])
        self.norm1 = nn.LayerNorm(cfg["d_model"])
        self.norm2 = nn.LayerNorm(cfg["d_model"])
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), cache)
        x = x + attn_out
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_cache


class CPTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.special = cfg["special_tokens"]

        vs = cfg["vocab_sizes"]
        es = cfg["token_embed_sizes"]
        ign = self.special["ignore_idx"]

        self.emb_family = nn.Embedding(vs["family"], es["family"], padding_idx=ign)
        self.emb_tempo = nn.Embedding(vs["tempo"], es["tempo"], padding_idx=ign)
        self.emb_pos = nn.Embedding(vs["position_bar"], es["position_bar"], padding_idx=ign)
        self.emb_pitch = nn.Embedding(vs["pitch"], es["pitch"], padding_idx=ign)
        self.emb_dur = nn.Embedding(vs["duration"], es["duration"], padding_idx=ign)
        self.emb_vel = nn.Embedding(vs["velocity"], es["velocity"], padding_idx=ign)

        total_emb = es["family"] + es["tempo"] + es["position_bar"] + es["pitch"] + es["duration"] + es["velocity"]
        self.w_in = nn.Linear(total_emb, cfg["d_model"], bias=False)

        self.abs_pos_emb = nn.Embedding(4096, cfg["d_model"])

        self.layers = nn.ModuleList([CPLayer(cfg) for _ in range(cfg["n_layers"])])
        self.norm = nn.LayerNorm(cfg["d_model"])

        self.w_f = nn.Linear(cfg["d_model"], vs["family"], bias=False)
        self.w_out_tempo = nn.Linear(cfg["d_model"] + es["family"], vs["tempo"], bias=False)
        self.w_out_pos = nn.Linear(cfg["d_model"] + es["family"], vs["position_bar"], bias=False)
        self.w_out_pitch = nn.Linear(cfg["d_model"] + es["family"], vs["pitch"], bias=False)
        self.w_out_dur = nn.Linear(cfg["d_model"] + es["family"], vs["duration"], bias=False)
        self.w_out_vel = nn.Linear(cfg["d_model"] + es["family"], vs["velocity"], bias=False)

    def forward(self, cp_seq, target_seq=None, cache=None, start_pos=0):
        B, T, _ = cp_seq.shape

        vs = self.cfg["vocab_sizes"]
        f_in  = torch.clamp(cp_seq[:, :, 0], max=vs["family"] - 1)
        t_in  = torch.clamp(cp_seq[:, :, 1], max=vs["tempo"] - 1)
        p_in  = torch.clamp(cp_seq[:, :, 2], max=vs["position_bar"] - 1)
        pi_in = torch.clamp(cp_seq[:, :, 3], max=vs["pitch"] - 1)
        d_in  = torch.clamp(cp_seq[:, :, 4], max=vs["duration"] - 1)
        v_in  = torch.clamp(cp_seq[:, :, 5], max=vs["velocity"] - 1)

        e_f  = self.emb_family(f_in)
        e_t  = self.emb_tempo(t_in)
        e_p  = self.emb_pos(p_in)
        e_pi = self.emb_pitch(pi_in)
        e_d  = self.emb_dur(d_in)
        e_v  = self.emb_vel(v_in)

        x = torch.cat([e_f, e_t, e_p, e_pi, e_d, e_v], dim=-1)
        h = self.w_in(x)

        t_pos = torch.arange(start_pos, start_pos + T, device=h.device)
        h = h + self.abs_pos_emb(t_pos)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, new_c = layer(h, layer_cache)
            new_caches.append(new_c)

        h = self.norm(h)

        logits_f = self.w_f(h)

        if target_seq is not None:
            gt_f = torch.clamp(target_seq[:, :, 0], max=vs["family"] - 1)
            e_f_stage2 = self.emb_family(gt_f)
        else:
            e_f_stage2 = e_f

        h_cond = torch.cat([h, e_f_stage2], dim=-1)

        logits_t  = self.w_out_tempo(h_cond)
        logits_p  = self.w_out_pos(h_cond)
        logits_pi = self.w_out_pitch(h_cond)
        logits_d  = self.w_out_dur(h_cond)
        logits_v  = self.w_out_vel(h_cond)

        if target_seq is not None:
            loss_dict = self.compute_loss(
                [logits_f, logits_t, logits_p, logits_pi, logits_d, logits_v],
                target_seq
            )
            return loss_dict

        out_logits = (logits_f, logits_t, logits_p, logits_pi, logits_d, logits_v)
        if cache is not None or not self.training:
            return out_logits, new_caches
        return out_logits

    def compute_loss(self, logits_list, target_seq):
        ign = self.special["ignore_idx"]
        pad = self.special["pad_idx"]
        bos = self.special["bos_family_idx"]
        vs  = self.cfg["vocab_sizes"]

        t_f  = torch.clamp(target_seq[:, :, 0].clone(), max=vs["family"] - 1)
        t_t  = torch.clamp(target_seq[:, :, 1].clone(), max=vs["tempo"] - 1)
        t_p  = torch.clamp(target_seq[:, :, 2].clone(), max=vs["position_bar"] - 1)
        t_pi = torch.clamp(target_seq[:, :, 3].clone(), max=vs["pitch"] - 1)
        t_d  = torch.clamp(target_seq[:, :, 4].clone(), max=vs["duration"] - 1)
        t_v  = torch.clamp(target_seq[:, :, 5].clone(), max=vs["velocity"] - 1)

        t_f[(t_f == ign) | (t_f == pad) | (t_f == bos)] = -100

        for t_x in [t_t, t_p, t_pi, t_d, t_v]:
            t_x[(t_x == ign) | (t_x == pad)] = -100

        ls = self.cfg["label_smoothing"]
        ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=ls)

        def safe_ce(logits, targets):
            if (targets != -100).any():
                return ce(logits.view(-1, logits.shape[-1]), targets.view(-1))
            return logits.sum() * 0.0

        l_f  = safe_ce(logits_list[0], t_f)
        l_t  = safe_ce(logits_list[1], t_t)
        l_p  = safe_ce(logits_list[2], t_p)
        l_pi = safe_ce(logits_list[3], t_pi)
        l_d  = safe_ce(logits_list[4], t_d)
        l_v  = safe_ce(logits_list[5], t_v)

        total = l_f + l_t + l_p + l_pi + l_d + l_v

        return {
            "total":        total,
            "family":       l_f,
            "tempo":        l_t,
            "position_bar": l_p,
            "pitch":        l_pi,
            "duration":     l_d,
            "velocity":     l_v
        }
