from typing import List, Optional
import os
import csv

import torch

from env.protein_env import ProteinEnv
from .protein_sequence_eval import compute_sequence_logprobs


def make_self_surprise_reward(
    trainer,
    tokenizer,
    env: ProteinEnv,
    aa_ids: List[int],
    id_eos: int,
    renorm_over_allowed: bool = True,
    base_ref_model=None,
    first_variation_coef: float = 0.0,
):
    """Create a reward function for TRL GRPO.

    Reward per sequence:
      R = -log p_ref(seq) - first_variation_coef * (log p_pol(seq) - log p_base(seq))

    Also logs in-update approximate KL: mean(log p_pol - log p_ref) per batch to
    outputs/grpo_approx_kl_in_update.csv.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    kl_path = os.path.join(out_dir, "grpo_approx_kl_in_update.csv")

    def _append_approx_kl(value: float) -> None:
        exists = os.path.exists(kl_path)
        with open(kl_path, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["approx_kl_in_update"])  # header
            w.writerow([float(value)])

    def _batched_reward(prompts, completion_ids) -> List[float]:
        ref = trainer.ref_model if getattr(trainer, "ref_model", None) is not None else trainer.model
        pol = trainer.model
        device = next(pol.parameters()).device

        # Tokenize prompts (no specials)
        prompt_ids_list: List[List[int]] = [
            tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts
        ]

        # Append EOS to completions if missing, build padded tensors
        seq_tensors: List[torch.Tensor] = []
        for pid, cid in zip(prompt_ids_list, completion_ids):
            seq_ids = pid + cid
            if len(seq_ids) == 0 or seq_ids[-1] != id_eos:
                seq_ids = seq_ids + [id_eos]
            seq_tensors.append(torch.tensor(seq_ids, dtype=torch.long))
        max_len = max(t.size(0) for t in seq_tensors)
        input_ids = torch.full((len(seq_tensors), max_len), fill_value=tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seq_tensors), max_len), dtype=torch.long)
        for i, t in enumerate(seq_tensors):
            input_ids[i, : t.size(0)] = t
            attention_mask[i, : t.size(0)] = 1  # length-based mask (handles PAD==EOS)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Sequence log-probs under ref and policy (legal-conditional if renorm_over_allowed=True)
        seq_logp_ref = compute_sequence_logprobs(
            ref, input_ids, attention_mask, tokenizer, env, aa_ids, id_eos, renorm_over_allowed=renorm_over_allowed
        )
        seq_logp_pol = compute_sequence_logprobs(
            pol, input_ids, attention_mask, tokenizer, env, aa_ids, id_eos, renorm_over_allowed=renorm_over_allowed
        )

        # Log approximate in-update KL on sampled sequences
        approx_kl_batch = (seq_logp_pol - seq_logp_ref).mean().item()
        _append_approx_kl(approx_kl_batch)

        # Reward = -log p_ref(seq) - coef * (log p_pol - log p_base)
        total = -seq_logp_ref
        if base_ref_model is not None and first_variation_coef != 0.0:
            base_ref_model.to(device)
            seq_logp_base = compute_sequence_logprobs(
                base_ref_model,
                input_ids,
                attention_mask,
                tokenizer,
                env,
                aa_ids,
                id_eos,
                renorm_over_allowed=renorm_over_allowed,
            )
            total = total - first_variation_coef * (seq_logp_pol - seq_logp_base)
        return total.float().cpu().tolist()

    def _single_reward(prompt: str, completion: str) -> float:
        ref = trainer.ref_model if getattr(trainer, "ref_model", None) is not None else trainer.model
        pol = trainer.model
        device = next(pol.parameters()).device
        pid = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        cid = tokenizer(completion, add_special_tokens=False)["input_ids"]
        seq_ids = pid + cid
        if len(seq_ids) == 0 or seq_ids[-1] != id_eos:
            seq_ids = seq_ids + [id_eos]
        input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        seq_logp_ref = compute_sequence_logprobs(
            ref, input_ids, attention_mask, tokenizer, env, aa_ids, id_eos, renorm_over_allowed=renorm_over_allowed
        )
        seq_logp_pol = compute_sequence_logprobs(
            pol, input_ids, attention_mask, tokenizer, env, aa_ids, id_eos, renorm_over_allowed=renorm_over_allowed
        )
        # KL logging (single sample)
        approx_kl = (seq_logp_pol - seq_logp_ref).item()
        _append_approx_kl(approx_kl)
        total = -seq_logp_ref
        if base_ref_model is not None and first_variation_coef != 0.0:
            base_ref_model.to(device)
            seq_logp_base = compute_sequence_logprobs(
                base_ref_model, input_ids, attention_mask, tokenizer, env, aa_ids, id_eos, renorm_over_allowed=renorm_over_allowed
            )
            total = total - first_variation_coef * (seq_logp_pol - seq_logp_base)
        return float(total.item())

    def reward_func(*args, **kwargs):
        # Per-sample kwargs
        prompt_kw: Optional[str] = kwargs.get("prompt")
        completion_kw: Optional[str] = kwargs.get("completion")
        if prompt_kw is not None and completion_kw is not None:
            return _single_reward(prompt_kw, completion_kw)

        # Batched via kwargs
        prompts_kw = kwargs.get("prompts")
        completion_ids_kw = kwargs.get("completion_ids") or kwargs.get("completion_ids_list")
        completions_kw = kwargs.get("completions")

        # Normalize completion_ids from tensors/tuples to Python lists of ints
        try:
            import torch as _torch
        except Exception:
            _torch = None

        def _to_py_id_lists(x):
            if x is None:
                return None
            if _torch is not None and isinstance(x, _torch.Tensor):
                return [list(map(int, row)) for row in x.detach().cpu().tolist()]
            if isinstance(x, (list, tuple)):
                out = []
                for r in x:
                    if _torch is not None and isinstance(r, _torch.Tensor):
                        out.append([int(t) for t in r.detach().cpu().tolist()])
                    elif isinstance(r, (list, tuple)):
                        out.append([int(t) for t in r])
                    elif isinstance(r, int):
                        out.append([int(r)])
                return out
            return None

        comp_ids_norm = _to_py_id_lists(completion_ids_kw)
        if isinstance(prompts_kw, (list, tuple)) and (comp_ids_norm is not None or isinstance(completions_kw, (list, tuple))):
            if comp_ids_norm is None and completions_kw is not None:
                comp_ids_norm = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in completions_kw]
            prompts_norm = list(prompts_kw)
            return _batched_reward(prompts_norm, comp_ids_norm)

        # Batched via positional args: (prompts, completions, completion_ids)
        if len(args) >= 3:
            prompts, completions, completion_ids = args[0], args[1], args[2]
            return _batched_reward(prompts, completion_ids)

        # Fallbacks to avoid None
        if isinstance(kwargs.get("prompt"), str) and isinstance(kwargs.get("completion"), str):
            return 0.0
        n = 1
        try:
            if len(args) >= 3 and isinstance(args[2], list):
                n = len(args[2])
            elif isinstance(kwargs.get("completion_ids"), list):
                n = len(kwargs.get("completion_ids"))
            elif isinstance(kwargs.get("completion_ids_list"), list):
                n = len(kwargs.get("completion_ids_list"))
            elif isinstance(kwargs.get("prompts"), list):
                n = len(kwargs.get("prompts"))
        except Exception:
            n = 1
        return [0.0] * n

    return reward_func


