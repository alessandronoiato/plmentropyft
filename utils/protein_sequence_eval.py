from typing import List, Tuple, Dict

import math
import torch

from env.protein_env import ProteinEnv
from .protein_validity import is_valid_basic


def compute_sequence_logprobs(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    tokenizer,
    env: ProteinEnv,
    aa_ids: List[int],
    id_eos: int,
    renorm_over_allowed: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        next_logits = logits[:, :-1, :]
        next_ids = input_ids[:, 1:]
        B, S, _ = next_logits.shape
        device = next_logits.device
        token_logps = torch.zeros((B, S), dtype=next_logits.dtype, device=device)
        special = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
        for b in range(B):
            seq_tokens = input_ids[b].tolist()
            id_to_len_cfg = getattr(getattr(env, "config", object()), "id_to_len", None)
            for t in range(S):
                prefix_ids = seq_tokens[: t + 1]
                prefix_actions = [i for i in prefix_ids if i not in special]
                # Letters vs pieces: if env provides id_to_len, compute allowed set directly for stability
                if isinstance(id_to_len_cfg, dict):
                    horizon = getattr(env.config, "horizon", 0)
                    consumed = sum(id_to_len_cfg.get(x, 0) for x in prefix_actions)
                    remaining = max(0, horizon - consumed)
                    if remaining == 0:
                        allowed_ids = [id_eos]
                    else:
                        allowed_ids = [tid for tid, length in id_to_len_cfg.items() if length <= remaining]
                else:
                    allowed_ids = env.legal_action_ids(prefix_actions, aa_ids, id_eos)  # type: ignore[arg-type]
                logits_bt = next_logits[b, t]
                if renorm_over_allowed:
                    if not allowed_ids:
                        continue
                    logZ = torch.logsumexp(logits_bt[allowed_ids], dim=-1)
                else:
                    logZ = torch.logsumexp(logits_bt, dim=-1)
                next_id = next_ids[b, t].item()
                # If pieces mode and budget exhausted but next token isn't EOS, stop scoring further steps for this sequence
                if isinstance(id_to_len_cfg, dict):
                    horizon = getattr(env.config, "horizon", 0)
                    consumed = sum(id_to_len_cfg.get(x, 0) for x in prefix_actions)
                    remaining = max(0, horizon - consumed)
                    if remaining == 0 and next_id != id_eos:
                        break
                logp = logits_bt[next_id] - logZ
                if attention_mask[b, t + 1].item() == 1:
                    token_logps[b, t] = logp
        return token_logps.sum(dim=-1)


def enumerate_sequence_probs(
    model, tokenizer, env: ProteinEnv, aa_ids: List[int], id_eos: int
) -> Tuple[float, List[Tuple[str, float]]]:
    device = next(model.parameters()).device
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    prefixes: List[Tuple[List[int], List[int], float]] = [([bos], [], 0.0)]
    finals: List[Tuple[str, float]] = []
    model.eval()
    aa_id_to_sym: Dict[int, str] = {}
    for tid in aa_ids:
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        aa_id_to_sym[tid] = tok[1:] if isinstance(tok, str) and tok.startswith("Ġ") else tok
    with torch.no_grad():
        while prefixes:
            next_prefixes = []
            for token_ids, actions, logp_sum in prefixes:
                id_to_len = getattr(getattr(env, "config", object()), "id_to_len", None)
                if isinstance(id_to_len, dict):
                    horizon = getattr(env.config, "horizon", 0)
                    consumed = sum(id_to_len.get(x, 0) for x in actions)
                    remaining = max(0, horizon - consumed)
                    if remaining == 0:
                        allowed_ids = [id_eos]
                    else:
                        allowed_ids = [tid for tid, length in id_to_len.items() if length <= remaining]
                else:
                    allowed_ids = env.legal_action_ids(actions, aa_ids, id_eos)  # type: ignore[arg-type]
                if not allowed_ids:
                    continue
                inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                attn = torch.ones_like(inp)
                logits_last = model(input_ids=inp, attention_mask=attn).logits[0, -1]
                logZ = torch.logsumexp(logits_last[allowed_ids], dim=-1).item()
                for tok in allowed_ids:
                    logp_tok = (logits_last[tok].item() - logZ)
                    new_logp = logp_sum + logp_tok
                    if tok == id_eos:
                        actions_str = "".join([aa_id_to_sym.get(a, "?") for a in actions])
                        finals.append((actions_str, math.exp(new_logp)))
                    else:
                        next_prefixes.append((token_ids + [tok], actions + [tok], new_logp))
            prefixes = next_prefixes
    total_p = sum(p for _, p in finals)
    if total_p > 0:
        finals = [(a, p / total_p) for a, p in finals]
    H = -sum(p * math.log(p + 1e-40) for _, p in finals)
    return H, finals


def compute_legal_mass_raw(
    model, tokenizer, env: ProteinEnv, aa_ids: List[int], id_eos: int
) -> float:
    device = next(model.parameters()).device
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    prefixes: List[Tuple[List[int], List[int], float]] = [([bos], [], 0.0)]
    total_mass = 0.0
    model.eval()
    with torch.no_grad():
        while prefixes:
            next_prefixes = []
            for token_ids, actions, logp_sum in prefixes:
                id_to_len = getattr(getattr(env, "config", object()), "id_to_len", None)
                if isinstance(id_to_len, dict):
                    horizon = getattr(env.config, "horizon", 0)
                    consumed = sum(id_to_len.get(x, 0) for x in actions)
                    remaining = max(0, horizon - consumed)
                    if remaining == 0:
                        allowed_ids = [id_eos]
                    else:
                        allowed_ids = [tid for tid, length in id_to_len.items() if length <= remaining]
                else:
                    allowed_ids = env.legal_action_ids(actions, aa_ids, id_eos)  # type: ignore[arg-type]
                if not allowed_ids:
                    continue
                inp = torch.tensor([token_ids], dtype=torch.long, device=device)
                attn = torch.ones_like(inp)
                logits_last = model(input_ids=inp, attention_mask=attn).logits[0, -1]
                logZ_full = torch.logsumexp(logits_last, dim=-1).item()
                for tok in allowed_ids:
                    logp_tok = logits_last[tok].item() - logZ_full
                    new_logp = logp_sum + logp_tok
                    if tok == id_eos:
                        total_mass += math.exp(new_logp)
                    else:
                        next_prefixes.append((token_ids + [tok], actions + [tok], new_logp))
            prefixes = next_prefixes
    return float(total_mass)


def actions_to_str(tokenizer, action_ids: List[int]) -> str:
    toks = tokenizer.convert_ids_to_tokens(action_ids)
    sym = [(t[1:] if isinstance(t, str) and t.startswith("Ġ") else t) for t in toks]
    return "".join(sym)


def sample_entropy_and_validity(
    model,
    tokenizer,
    horizon: int,
    num_samples: int,
    *,
    do_sample: bool = True,
    top_p: float = 1.0,
    top_k: int = 0,
    temperature: float = 1.0,
) -> Tuple[float, List[Tuple[str, float]], float, List[Tuple[str, float]]]:
    model_dev = next(model.parameters()).device
    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    queries = torch.tensor([[start_id]] * num_samples, dtype=torch.long, device=model_dev)
    from transformers import GenerationConfig

    gen_cfg = GenerationConfig(
        max_new_tokens=horizon + 1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    attn_mask = torch.ones_like(queries)
    out = model.generate(inputs=queries, attention_mask=attn_mask, generation_config=gen_cfg, return_dict_in_generate=True)
    responses = out.sequences.to(model_dev)
    # Build attention mask by true sequence length up to first EOS (include EOS)
    attn = torch.zeros_like(responses, dtype=torch.long, device=model_dev)
    eos_tok = tokenizer.eos_token_id
    for i in range(responses.size(0)):
        row = responses[i].tolist()
        try:
            first_eos = row.index(eos_tok, 1)
        except ValueError:
            first_eos = len(row) - 1
        attn[i, : first_eos + 1] = 1
    # Build empirical distribution by residue strings using counts
    special_ids = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    seq_counts: Dict[str, int] = {}
    validity_sum = 0.0
    residue_strs: List[str] = []
    per_sample_validity: List[Tuple[str, float]] = []
    total_token_len_pre_eos = 0.0
    total_residue_len_raw = 0.0
    for i in range(responses.size(0)):
        ids = responses[i].tolist()
        # Determine first EOS to measure token length and build raw (pre-trunc) residue string
        try:
            first_eos_i = ids.index(eos_tok, 1)
        except ValueError:
            first_eos_i = len(ids) - 1
        token_len_pre_eos = max(first_eos_i - 1, 0)  # exclude BOS and EOS
        total_token_len_pre_eos += float(token_len_pre_eos)
        action_ids_raw = [tid for tid in ids[1:first_eos_i] if tid not in special_ids]
        s_raw = actions_to_str(tokenizer, action_ids_raw)
        total_residue_len_raw += float(len(s_raw))
        s = s_raw
        # No truncation/padding: validity and CSV are based on full raw decoded residues
        seq_counts[s] = seq_counts.get(s, 0) + 1
        v01 = float(is_valid_basic(s, min_len=1, max_len=10_000, allow_U=False, allow_Cdot=True, max_run=6))
        validity_sum += v01
        per_sample_validity.append((s, v01))
        residue_strs.append(s)
    N = sum(seq_counts.values())
    seq_list: List[Tuple[str, float]] = []
    if N > 0:
        for s, c in seq_counts.items():
            p = c / N
            seq_list.append((s, p))
    # Monte Carlo entropy estimator on sampled token paths only (no padding in NLL)
    # H ≈ -(1/N) * sum_i log q(path_i), where path_i is the generated tokenization up to EOS
    H_hat = float("nan")
    with torch.no_grad():
        logits = model(input_ids=responses, attention_mask=attn).logits
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        targets = responses[:, 1:]
        tgt_mask = attn[:, 1:]
        token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * tgt_mask
        seq_logps = token_logps.sum(dim=1)
        H_hat = float((-seq_logps).mean().item())
    mean_validity = float(validity_sum / max(1, N))
    mean_token_len = float(total_token_len_pre_eos / max(1, N))
    mean_residue_len = float(total_residue_len_raw / max(1, N))
    return H_hat, sorted(seq_list, key=lambda x: -x[1]), mean_validity, per_sample_validity, mean_token_len, mean_residue_len


