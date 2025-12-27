"""
SGPO Pipeline and Reward Function

This module provides the complete SGPO pipeline for generating, aligning,
projecting, and scoring protein sequences, along with the reward function
for GRPO training.
"""

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import MAFFTAligner
from .model import ProGen2Wrapper
from .oracle import SGPOFitnessOracle
from .projector import TrpBProjector


def compute_sequence_logprobs_simple(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
) -> torch.Tensor:
    """
    Compute sequence log-probabilities under a model.
    
    This is a simplified version that doesn't do legal-action renormalization,
    suitable for ProGen2 and similar models.
    
    Args:
        model: Language model
        input_ids: (B, S) token IDs
        attention_mask: (B, S) attention mask
    
    Returns:
        (B,) tensor of sequence log-probabilities
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for next-token prediction: predict token t+1 from position t
        shift_logits = logits[:, :-1, :]  # (B, S-1, V)
        shift_labels = input_ids[:, 1:]    # (B, S-1)
        shift_mask = attention_mask[:, 1:] # (B, S-1)
        
        # Compute log-softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, S-1, V)
        
        # Gather log-probs of actual next tokens
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, S-1)
        
        # Mask out padding positions
        token_log_probs = token_log_probs * shift_mask.float()
        
        # Sum to get sequence log-prob
        seq_log_probs = token_log_probs.sum(dim=-1)  # (B,)
        
        return seq_log_probs


class SGPOPipeline:
    """
    Complete SGPO pipeline: generate -> align -> project -> score.
    
    This encapsulates the full workflow from SGPO for generating sequences,
    aligning them to the parent, projecting to combo format, and scoring.
    """
    
    def __init__(
        self,
        model: ProGen2Wrapper,
        aligner: MAFFTAligner,
        projector: TrpBProjector,
        oracle: SGPOFitnessOracle,
    ):
        """
        Initialize pipeline.
        
        Args:
            model: ProGen2 model wrapper
            aligner: MAFFT alignment wrapper
            projector: TrpB sequence projector
            oracle: Fitness oracle
        """
        self.model = model
        self.aligner = aligner
        self.projector = projector
        self.oracle = oracle
    
    def generate_and_score(
        self,
        num_sequences: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        batch_size: int = 40,
    ) -> Dict[str, Any]:
        """
        Generate sequences and score them.
        
        Args:
            num_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            batch_size: Generation batch size
        
        Returns:
            Dictionary with:
                - full_sequences: Generated full sequences
                - aligned_sequences: MAFFT-aligned sequences
                - projected_sequences: Projected sequences
                - combos: 15-char combo sequences
                - fitness_scores: Oracle fitness scores
        """
        all_full = []
        all_aligned = []
        all_projected = []
        all_combos = []
        all_scores = []
        
        remaining = num_sequences
        while remaining > 0:
            n = min(batch_size, remaining)
            
            # Generate
            generated = self.model.sample(
                num_return_sequences=n,
                temperature=temperature,
                top_p=top_p,
            )
            
            if not generated:
                remaining -= n
                continue
            
            # Align
            aligned = self.aligner.align(generated)
            
            # Project
            projected, combos = self.projector.project(aligned)
            
            # Score
            scores = self.oracle(combos)
            
            all_full.extend(generated)
            all_aligned.extend(aligned)
            all_projected.extend(projected)
            all_combos.extend(combos)
            all_scores.extend(scores)
            
            remaining -= n
        
        return {
            "full_sequences": all_full,
            "aligned_sequences": all_aligned,
            "projected_sequences": all_projected,
            "combos": all_combos,
            "fitness_scores": all_scores,
        }
    
    def score_sequences(self, sequences: List[str]) -> Tuple[List[float], List[str]]:
        """
        Score pre-generated sequences.
        
        Args:
            sequences: List of full protein sequences
        
        Returns:
            (fitness_scores, combos)
        """
        if not sequences:
            return [], []
        
        # Align and project
        aligned = self.aligner.align(sequences)
        _, combos = self.projector.project(aligned)
        scores = self.oracle(combos)
        
        return scores, combos


def make_fitness_reward(
    pipeline: SGPOPipeline,
    trainer,  # GRPOTrainer - provides ref_model and model
    tokenizer,
    entropy_coef: float = 1.0,
    first_variation_coef: float = 0.0,
    fitness_scale: float = 1.0,
    base_ref_model=None,  # Optional base model for first variation KL term
    out_dir: Optional[str] = None,
):
    """
    Create a reward function that combines fitness with entropy regularization.
    
    Reward per sequence (first variation of J = E[λr] + μ H - η KL):
        R(y) = λ * r(y) - (μ + η) * log p_ref(y) + η * log p_base(y)
    
    where:
        λ = fitness_scale (external fitness reward weight)
        μ = entropy_coef (entropy bonus)
        η = first_variation_coef (KL-to-base penalty)
        r(y) = fitness oracle score
        p_ref = reference policy (frozen copy of policy at start)
        p_base = base model (optional, for anchoring to pretrained)
    
    Args:
        pipeline: SGPOPipeline for align -> project -> score
        trainer: GRPOTrainer instance (provides ref_model and model)
        tokenizer: Tokenizer for decoding sequences
        entropy_coef: μ (entropy coefficient)
        first_variation_coef: η (KL-to-base coefficient)
        fitness_scale: λ (scale factor for fitness reward)
        base_ref_model: Optional base model for first variation term
        out_dir: Directory for logging
    
    Returns:
        Reward function compatible with TRL's GRPO trainer
    """
    
    # Setup logging
    out_dir_path = out_dir or "."
    os.makedirs(out_dir_path, exist_ok=True)
    kl_path = os.path.join(out_dir_path, "grpo_approx_kl_in_update.csv")
    fv_path = os.path.join(out_dir_path, "grpo_first_variation_in_update.csv")
    
    # Logging state
    _log_state = {"batch_count": 0, "total_scored": 0}
    _fitness_log: List[Dict[str, Any]] = []
    
    def _append_approx_kl(value: float) -> None:
        exists = os.path.exists(kl_path)
        with open(kl_path, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["approx_kl_in_update"])
            w.writerow([float(value)])
    
    def _append_first_variation(
        batch_mean_logp_ref: float, 
        batch_mean_logp_base: float, 
        coef: float,
        batch_mean_fitness: float,
    ) -> None:
        exists = os.path.exists(fv_path)
        with open(fv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow([
                    "mean_logp_ref", "mean_logp_base", 
                    "first_variation_coef", "penalty_value",
                    "mean_fitness"
                ])
            penalty = coef * (batch_mean_logp_ref - batch_mean_logp_base)
            w.writerow([
                float(batch_mean_logp_ref), 
                float(batch_mean_logp_base), 
                float(coef), 
                float(penalty),
                float(batch_mean_fitness),
            ])
    
    def _decode_to_aa(token_ids) -> str:
        """Decode token IDs to amino acid sequence."""
        if hasattr(tokenizer, 'decode'):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        elif hasattr(tokenizer, 'decode_batch'):
            text = tokenizer.decode(token_ids)
        else:
            text = str(token_ids)
        aa_seq = "".join(c for c in text.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        return aa_seq
    
    def _batched_reward(prompts, completion_ids) -> List[float]:
        # Get models from trainer
        ref = getattr(trainer, "ref_model", None) or trainer.model
        pol = trainer.model
        device = next(pol.parameters()).device
        
        # Build input tensors from prompts + completions
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0) or 0
        
        # Tokenize prompts
        prompt_ids_list: List[List[int]] = []
        for p in prompts:
            if hasattr(tokenizer, 'encode'):
                pids = tokenizer.encode(p, add_special_tokens=False)
            else:
                pids = tokenizer(p, add_special_tokens=False)["input_ids"]
            prompt_ids_list.append(pids if isinstance(pids, list) else list(pids))
        
        # Build full sequences (prompt + completion)
        seq_tensors: List[torch.Tensor] = []
        for pid, cid in zip(prompt_ids_list, completion_ids):
            if isinstance(cid, torch.Tensor):
                cid = cid.cpu().tolist()
            seq_ids = list(pid) + list(cid)
            seq_tensors.append(torch.tensor(seq_ids, dtype=torch.long))
        
        # Pad to same length
        max_len = max(t.size(0) for t in seq_tensors)
        input_ids = torch.full(
            (len(seq_tensors), max_len), 
            fill_value=pad_token_id, 
            dtype=torch.long
        )
        attention_mask = torch.zeros((len(seq_tensors), max_len), dtype=torch.long)
        for i, t in enumerate(seq_tensors):
            input_ids[i, :t.size(0)] = t
            attention_mask[i, :t.size(0)] = 1
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Compute sequence log-probs under ref and policy
        seq_logp_ref = compute_sequence_logprobs_simple(ref, input_ids, attention_mask)
        seq_logp_pol = compute_sequence_logprobs_simple(pol, input_ids, attention_mask)
        
        # Log approximate in-update KL
        approx_kl_batch = (seq_logp_pol - seq_logp_ref).mean().item()
        _append_approx_kl(approx_kl_batch)
        
        # Decode completions to AA sequences for fitness scoring
        sequences = []
        for cid in completion_ids:
            if isinstance(cid, torch.Tensor):
                cid = cid.cpu().tolist()
            aa_seq = _decode_to_aa(cid)
            sequences.append(aa_seq)
        
        # DEBUG: Log sample sequences on first few batches
        if _log_state["batch_count"] < 3:
            print(f"\n[DEBUG batch {_log_state['batch_count']+1}] Sample decoded sequences:")
            for i, seq in enumerate(sequences[:3]):
                print(f"  Seq {i}: len={len(seq)}, first50='{seq[:50]}'")
        
        # Get fitness scores from oracle
        fitness_scores, combos = pipeline.score_sequences(sequences)
        
        # DEBUG: Log sample combos and scores on first few batches
        if _log_state["batch_count"] < 3:
            print(f"[DEBUG batch {_log_state['batch_count']+1}] Sample combos and scores:")
            for i, (combo, score) in enumerate(zip(combos[:3], fitness_scores[:3])):
                print(f"  Combo {i}: '{combo}' (len={len(combo)}) -> fitness={score:.4f}")
        fitness_tensor = torch.tensor(fitness_scores, dtype=torch.float32, device=device)
        
        # Compute reward: R = λ*fitness - (μ + η)*log p_ref + η*log p_base
        # where μ = entropy_coef, η = first_variation_coef
        
        if base_ref_model is not None and first_variation_coef != 0.0:
            # Move base model to device and compute its log-probs
            base_ref_model.to(device)
            seq_logp_base = compute_sequence_logprobs_simple(
                base_ref_model, input_ids, attention_mask
            )
            
            # Log first variation stats
            _append_first_variation(
                batch_mean_logp_ref=float(seq_logp_ref.mean().item()),
                batch_mean_logp_base=float(seq_logp_base.mean().item()),
                coef=float(first_variation_coef),
                batch_mean_fitness=float(fitness_tensor.mean().item()),
            )
            
            # Full first variation reward
            total = (
                fitness_scale * fitness_tensor 
                - (entropy_coef + first_variation_coef) * seq_logp_ref 
                + first_variation_coef * seq_logp_base
            )
        else:
            # No base model or η=0: just fitness + entropy term
            total = fitness_scale * fitness_tensor - entropy_coef * seq_logp_ref
        
        rewards = total.float().cpu().tolist()
        
        # Logging
        _log_state["batch_count"] += 1
        _log_state["total_scored"] += len(sequences)
        
        if fitness_scores:
            _fitness_log.append({
                "batch": _log_state["batch_count"],
                "mean_fitness": float(np.mean(fitness_scores)),
                "max_fitness": float(np.max(fitness_scores)),
                "min_fitness": float(np.min(fitness_scores)),
                "mean_reward": float(np.mean(rewards)),
                "mean_logp_ref": float(seq_logp_ref.mean().item()),
                "n_sequences": len(sequences),
            })
            
            # Periodically save log
            if out_dir and _log_state["batch_count"] % 10 == 0:
                log_path = os.path.join(out_dir, "fitness_log.json")
                try:
                    with open(log_path, "w") as f:
                        json.dump(_fitness_log, f, indent=2)
                except Exception:
                    pass
        
        return rewards
    
    def reward_func(*args, **kwargs):
        # Handle various calling conventions from TRL
        prompts_kw = kwargs.get("prompts")
        completion_ids_kw = kwargs.get("completion_ids") or kwargs.get("completion_ids_list")
        
        if prompts_kw is not None and completion_ids_kw is not None:
            # Normalize to Python lists
            def _to_py_lists(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return [list(map(int, row)) for row in x.detach().cpu().tolist()]
                if isinstance(x, (list, tuple)):
                    out = []
                    for r in x:
                        if isinstance(r, torch.Tensor):
                            out.append([int(t) for t in r.detach().cpu().tolist()])
                        elif isinstance(r, (list, tuple)):
                            out.append([int(t) for t in r])
                    return out
                return None
            
            comp_ids = _to_py_lists(completion_ids_kw)
            return _batched_reward(list(prompts_kw), comp_ids)
        
        # Fallback
        return [0.0]
    
    # Attach log accessor
    reward_func.get_fitness_log = lambda: _fitness_log
    
    return reward_func

