import argparse
import os
import sys
import json
import math
import csv
import random
import numpy as np
from typing import List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.trainer_callback import TrainerCallback

try:
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.grpo_config import GRPOConfig
except Exception as _:
    GRPOTrainer = None
    GRPOConfig = None


# Ensure project root on sys.path when running as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.protein_env import ProteinEnv, ProteinConfig
from utils.token_utils import get_amino_acid_token_ids
from utils.protein_sequence_eval import sample_entropy_and_validity
from utils.protein_validity import is_valid_basic
from utils.protein_reward import make_self_surprise_reward
from utils.protein_sequence_distance import topk_distance_avg
from utils.wandb_logger import maybe_init_wandb, log_report, finish as wandb_finish


def get_preferred_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nferruz/ProtGPT2")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.05, help="KL coefficient for GRPO")
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt (must divide batch)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_samples", type=int, default=128)
    parser.add_argument("--eval_do_sample", dest="eval_do_sample", action="store_true", default=True)
    parser.add_argument("--no_eval_do_sample", dest="eval_do_sample", action="store_false")
    parser.add_argument("--eval_top_p", type=float, default=1.0)
    parser.add_argument("--eval_top_k", type=int, default=0)
    parser.add_argument("--eval_temperature", type=float, default=1.0)
    parser.add_argument("--first_variation_coef", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default=os.path.join(_PROJECT_ROOT, "outputs"))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(_PROJECT_ROOT, "hf_cache"))
    parser.add_argument("--local_files_only", action="store_true", default=False)
    parser.add_argument("--enumerate_max_horizon", type=int, default=3, help="(deprecated) kept for compatibility; enumeration disabled")
    parser.add_argument("--tokenizer_mode", type=str, default="letters", choices=["letters", "pieces"], help="(deprecated) kept for compatibility")
    # Validity mode and ESMFold controls
    parser.add_argument("--validity_mode", type=str, default="basic", choices=["basic", "esmfold"], help="Choose validity oracle (mutually exclusive)")
    parser.add_argument("--prefilter_mode", type=str, default="none", choices=["none", "basic", "esmfold"], help="Optional prefilter; default none")
    parser.add_argument("--fold_plddt_threshold", type=float, default=70.0)
    parser.add_argument("--fold_device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--fold_batch_size", type=int, default=1)
    parser.add_argument("--fold_cache_dir", type=str, default=None)
    parser.add_argument("--eval_samples_fold_max", type=int, default=None)
    # Optional Vendi diversity (ESM2 embeddings)
    parser.add_argument("--compute_vendi", action="store_true", default=False)
    parser.add_argument("--vendi_model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--vendi_device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--vendi_batch_size", type=int, default=16)
    parser.add_argument("--vendi_kernel", type=str, default="cosine", choices=["cosine", "rbf", "linear"])
    parser.add_argument("--vendi_sigma", type=float, default=None)
    parser.add_argument("--vendi_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    # Pairwise top-k%% distance metric (evaluation)
    parser.add_argument("--pairwise_distance_mode", type=str, default="global", choices=["global", "hamming"], help="Distance mode: global (Needleman–Wunsch) or hamming (ungapped)")
    parser.add_argument("--pairwise_topk_percent", type=int, default=5, help="Top k percent of largest distances to average")
    parser.add_argument("--pairwise_num_pairs", type=int, default=5000, help="Number of random pairs to sample for distance computation")
    parser.add_argument("--pairwise_seed", type=int, default=123, help="Random seed for pair sampling")
    parser.add_argument("--pairwise_gap_penalty", type=int, default=1, help="Gap penalty for Needleman–Wunsch (global) distance")
    parser.add_argument("--pairwise_validity_filter", type=str, default="none", choices=["none", "basic", "esmfold"], help="Validity oracle for distance metric filtering")
    parser.add_argument("--pairwise_valid_strategy", type=str, default="collect_until", choices=["collect_until", "filter_after"], help="Valid filtering strategy: collect_until (Option 1a) or filter_after (Option 2)")
    parser.add_argument("--pairwise_collect_max_rounds", type=int, default=10, help="Max rounds of sampling when using collect_until")
    parser.add_argument("--pairwise_eval_budget", type=int, default=None, help="If set, filter_after generates/folds exactly this many sequences per side")
    parser.add_argument("--pairwise_eval_chunk_size", type=int, default=None, help="Optional chunk size for filter_after when pairwise_eval_budget is set")
    # Optional Weights & Biases logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline"])
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--no_json_report", action="store_true", help="Skip writing grpo_exact_entropy.json if set")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="Optional WandB API key for programmatic login on compute nodes")
    args = parser.parse_args()

    if GRPOTrainer is None or GRPOConfig is None:
        raise RuntimeError("trl[grpo] is required. Install a version that provides GRPOTrainer and GRPOConfig.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Prefer NVIDIA GPU if available, otherwise CPU
    device = get_preferred_device()
    # Do not set a global default device; Accelerate/TRL manages device placement.

    # Tokenizer and models
    # If a local path is provided, force offline load
    is_local_model = os.path.isdir(args.model_id)
    local_only = args.local_files_only or is_local_model or os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=local_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Decoder-only models should use left padding for generation
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    eos_id = tok.eos_token_id

    # Amino-acid token ids (letters mode only; pieces deprecated here)
    aa_ids = get_amino_acid_token_ids(tok)
    assert len(aa_ids) == 20 and len(set(aa_ids)) == 20, "AA id extraction must return exactly 20 unique ids"

    # Environment (unused for generation; kept for reward compatibility)
    env = ProteinEnv(ProteinConfig(horizon=args.horizon))

    # Policy and ref models
    policy = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=local_only)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=local_only)
    ref_model.requires_grad_(False)

    # Frozen pretrained base policy π0 for first-variation term
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=local_only)
    base_model.requires_grad_(False)
    base_model.to(device)

    # Save policy to a local directory so GRPO can reload both policy and ref from the same model path
    artifacts_dir = os.path.join(_PROJECT_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "prot_grpo_policy_init")
    policy.save_pretrained(model_path)
    tok.save_pretrained(model_path)

    # Keep a frozen handle to the initial (pre-training) policy for "before" evaluations
    before_model_for_eval = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=args.cache_dir, local_files_only=local_only
    )
    before_model_for_eval.requires_grad_(False)
    before_model_for_eval.eval()

    # Sanity check removed: no legality constraints in generation

    # Build GRPO config
    if args.batch_size % args.num_generations != 0:
        raise ValueError("batch_size must be divisible by num_generations for GRPO.")

    grpo_cfg = GRPOConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.steps,
        seed=args.seed,
        fp16=False,
        bf16=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        repetition_penalty=1.0,
        max_completion_length=args.horizon + 1,
        num_generations=args.num_generations,
        steps_per_generation=None,
        generation_batch_size=None,
        beta=args.beta,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=1.0,
        remove_unused_columns=False,
        report_to=[],
        logging_strategy="steps",
        logging_steps=1,
        use_transformers_paged=False,
        use_vllm=False,
        scale_rewards="none"
    )

    # Dataset of prompts: use explicit BOS token to guarantee non-empty inputs
    prompts = [tok.bos_token or "<|bos|>"] * args.batch_size
    train_ds = Dataset.from_dict({"prompt": prompts})

    # Reward function placeholder (uses fixed ref until trainer is constructed)
    reward_fn = make_self_surprise_reward(
        trainer=None,  # type: ignore[arg-type]
        tokenizer=tok,
        env=env,
        aa_ids=aa_ids,
        id_eos=eos_id,
        renorm_over_allowed=False,
        base_ref_model=base_model,
        first_variation_coef=args.first_variation_coef,
        out_dir=args.out_dir,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        processing_class=tok,
    )

    # No masked generation: legality constraints removed

    # Replace reward to dynamically refer to trainer.ref_model after init
    trainer.reward_funcs = [
        make_self_surprise_reward(
            trainer,
            tok,
            env,
            aa_ids,
            eos_id,
            renorm_over_allowed=False,
            base_ref_model=base_model,
            first_variation_coef=args.first_variation_coef,
            out_dir=args.out_dir,
        )
    ]
    trainer.reward_func_names = ["self_surprise_ref_k"]

    # No exact entropy callback: we use Monte Carlo estimates only

    # BEFORE distribution (initial policy)
    # Optional WandB init
    wandb_run = None
    try:
        wandb_config = {
            "model_id": args.model_id,
            "horizon": args.horizon,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "num_generations": args.num_generations,
            "beta": args.beta,
            "first_variation_coef": args.first_variation_coef,
            "validity_mode": args.validity_mode,
            "pairwise_distance_mode": args.pairwise_distance_mode,
            "pairwise_topk_percent": args.pairwise_topk_percent,
            "pairwise_num_pairs": args.pairwise_num_pairs,
            "pairwise_validity_filter": args.pairwise_validity_filter,
            "pairwise_valid_strategy": args.pairwise_valid_strategy,
        }
        wandb_run = maybe_init_wandb(args, wandb_config)
    except Exception:
        wandb_run = None
    def _reseed_all(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
        random.seed(seed)
        np.random.seed(seed)
    def dump_sequence_probs(model, csv_path, validity_csv_path):
        # Monte Carlo estimate of distribution and entropy without legality constraints
        # Determine fold device
        fold_device = args.fold_device
        if fold_device == "auto":
            fold_device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
        H, H_valid, seqs, mean_valid, per_valid, mean_token_len, mean_residue_len = sample_entropy_and_validity(
            model,
            tok,
            args.horizon,
            max(args.eval_samples, args.batch_size),
            do_sample=args.eval_do_sample,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            temperature=args.eval_temperature,
            validity_mode=args.validity_mode,
            fold_device=fold_device,
            fold_batch_size=args.fold_batch_size,
            fold_plddt_threshold=args.fold_plddt_threshold,
            eval_samples_fold_max=args.eval_samples_fold_max,
            fold_cache_dir=args.fold_cache_dir,
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "probability"])  # empirical from samples
            for a, p in seqs:
                w.writerow([a, p])
        with open(validity_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            # Header depends on validity mode
            if args.validity_mode == "esmfold":
                w.writerow(["sequence", "valid", "plddt_mean", "plddt_median", "fold_ok", "fold_error"])  # esmfold stats
                for rec in per_valid:
                    w.writerow([
                        rec.get("sequence", ""),
                        int(rec.get("valid", 0)),
                        rec.get("plddt_mean"),
                        rec.get("plddt_median"),
                        rec.get("fold_ok"),
                        rec.get("fold_error"),
                    ])
            else:
                w.writerow(["sequence", "valid"])  # basic oracle
                for rec in per_valid:
                    w.writerow([rec.get("sequence", ""), int(rec.get("valid", 0))])
        return H, H_valid, seqs, mean_valid, mean_token_len, mean_residue_len

    # Reseed to ensure matched RNG start for the BEFORE sampling pass
    _reseed_all(args.seed)
    model_before = trainer.accelerator.unwrap_model(trainer.model)
    before_csv = os.path.join(args.out_dir, "before_sequence_probs.csv")
    before_valid_csv = os.path.join(args.out_dir, "before_validity.csv")
    H_before, H_before_valid, seqs_before, V_before, Ltok_before, Lres_before = dump_sequence_probs(model_before, before_csv, before_valid_csv)

    # Save exact entropy before finetuning if available
    with open(os.path.join(args.out_dir, "before_exact_entropy.json"), "w") as f:
        json.dump({"entropy_nats": float(H_before), "mean_validity": float(V_before), "num_sequences": len(seqs_before)}, f, indent=2)

    # Train
    trainer.train()

    # After: Monte Carlo estimate on the fine-tuned policy
    with torch.no_grad():
        # Reseed to ensure matched RNG start for the AFTER sampling pass
        _reseed_all(args.seed)
        model_eval = trainer.accelerator.unwrap_model(trainer.model)
        after_csv = os.path.join(args.out_dir, "after_sequence_probs.csv")
        after_valid_csv = os.path.join(args.out_dir, "after_validity.csv")
        H_after, H_after_valid, seqs_after, V_after, Ltok_after, Lres_after = dump_sequence_probs(model_eval, after_csv, after_valid_csv)

    report = {
        "horizon": args.horizon,
        "num_sequences_before": len(seqs_before),
        "num_sequences_after": len(seqs_after),
        "before_entropy_nats": float(H_before),
        "after_entropy_nats": float(H_after),
        "before_entropy_nats_valid_only": float(H_before_valid),
        "after_entropy_nats_valid_only": float(H_after_valid),
        "before_mean_validity": float(V_before),
        "after_mean_validity": float(V_after),
        "before_mean_token_length_to_eos": float(Ltok_before),
        "after_mean_token_length_to_eos": float(Ltok_after),
        "before_mean_residue_length_to_eos": float(Lres_before),
        "after_mean_residue_length_to_eos": float(Lres_after),
        "mean_token_length_delta": float(Ltok_after - Ltok_before),
        "mean_residue_length_delta": float(Lres_after - Lres_before),
        # theoretical_max_nats removed: not applicable to BPE token-level MC NLL
        "sum_probs_before": sum(p for _, p in seqs_before) if seqs_before else float("nan"),
        "sum_probs_after": sum(p for _, p in seqs_after) if seqs_after else float("nan"),
    }
    # Pairwise top-k% distance metric (before/after)
    try:
        from math import ceil, sqrt

        def _compute_topk_for_sequences(seq_list):
            return topk_distance_avg(
                seq_list,
                mode=args.pairwise_distance_mode,
                topk_percent=args.pairwise_topk_percent,
                num_pairs=args.pairwise_num_pairs,
                seed=args.pairwise_seed,
                gap_penalty=args.pairwise_gap_penalty,
            )

        # Extract sequences as generated
        seqs_only_b_all = [a for a, _ in seqs_before]
        seqs_only_a_all = [a for a, _ in seqs_after]

        # Helper to filter valid sequences from a batch of per_valid records
        def _filter_valid_from_records(records):
            out = []
            for rec in records:
                try:
                    if int(rec.get("valid", 0)) == 1 and isinstance(rec.get("sequence"), str) and len(rec.get("sequence")) > 0:
                        out.append(rec.get("sequence"))
                except Exception:
                    continue
            return out

        # Validity filtering strategies
        valid_filter = str(args.pairwise_validity_filter)
        valid_strategy = str(args.pairwise_valid_strategy)

        before_valid_count = None
        after_valid_count = None
        before_feasible = None
        after_feasible = None

        seqs_only_b = list(seqs_only_b_all)
        seqs_only_a = list(seqs_only_a_all)

        if valid_filter != "none":
            # Determine required number of valid sequences for target pairs
            n_valid_min = int(ceil((1.0 + sqrt(1.0 + 8.0 * float(args.pairwise_num_pairs))) / 2.0))

            if valid_strategy == "filter_after":
                # Use per_valid from initial eval to filter
                # Re-run sampling to collect per_valid for before and after using the same settings
                def _collect_valid_sequences(model):
                    fold_device = args.fold_device
                    if fold_device == "auto":
                        fold_device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
                    budget = int(getattr(args, "pairwise_eval_budget", 0) or 0)
                    if budget and budget > 0:
                        chunk_size = int(getattr(args, "pairwise_eval_chunk_size", 0) or 0)
                        if chunk_size <= 0:
                            chunk_size = min(int(args.eval_samples), budget)
                        remaining = budget
                        valids_acc = []
                        while remaining > 0:
                            this_chunk = min(chunk_size, remaining)
                            _, _, seqs, _, per_valid, _, _ = sample_entropy_and_validity(
                                model,
                                tok,
                                args.horizon,
                                this_chunk,
                                do_sample=args.eval_do_sample,
                                top_p=args.eval_top_p,
                                top_k=args.eval_top_k,
                                temperature=args.eval_temperature,
                                validity_mode=args.validity_mode if valid_filter == "esmfold" else "basic",
                                fold_device=fold_device,
                                fold_batch_size=args.fold_batch_size,
                                fold_plddt_threshold=args.fold_plddt_threshold,
                                eval_samples_fold_max=(this_chunk if valid_filter == "esmfold" else args.eval_samples_fold_max),
                                fold_cache_dir=args.fold_cache_dir,
                            )
                            valids_acc.extend(_filter_valid_from_records(per_valid))
                            remaining -= this_chunk
                        return valids_acc
                    else:
                        gen_count = max(args.eval_samples, args.batch_size)
                        fold_cap = args.eval_samples_fold_max
                        _, _, seqs, _, per_valid, _, _ = sample_entropy_and_validity(
                            model,
                            tok,
                            args.horizon,
                            gen_count,
                            do_sample=args.eval_do_sample,
                            top_p=args.eval_top_p,
                            top_k=args.eval_top_k,
                            temperature=args.eval_temperature,
                            validity_mode=args.validity_mode if valid_filter == "esmfold" else "basic",
                            fold_device=fold_device,
                            fold_batch_size=args.fold_batch_size,
                            fold_plddt_threshold=args.fold_plddt_threshold,
                            eval_samples_fold_max=fold_cap,
                            fold_cache_dir=args.fold_cache_dir,
                        )
                        return _filter_valid_from_records(per_valid)

                seqs_only_b = _collect_valid_sequences(before_model_for_eval)
                seqs_only_a = _collect_valid_sequences(trainer.accelerator.unwrap_model(trainer.model))
                before_valid_count = len(seqs_only_b)
                after_valid_count = len(seqs_only_a)
                before_feasible = before_valid_count >= n_valid_min
                after_feasible = after_valid_count >= n_valid_min
            else:
                # collect_until: accumulate valid sequences across rounds until n_valid_min
                def _accumulate_valid_sequences(model):
                    fold_device = args.fold_device
                    if fold_device == "auto":
                        fold_device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
                    acc = []
                    seen = set()
                    max_rounds = int(getattr(args, "pairwise_collect_max_rounds", 10))
                    rounds = 0
                    while len(acc) < n_valid_min and rounds < max_rounds:
                        _, _, seqs, _, per_valid, _, _ = sample_entropy_and_validity(
                            model,
                            tok,
                            args.horizon,
                            max(args.eval_samples, args.batch_size),
                            do_sample=args.eval_do_sample,
                            top_p=args.eval_top_p,
                            top_k=args.eval_top_k,
                            temperature=args.eval_temperature,
                            validity_mode=args.validity_mode if valid_filter == "esmfold" else "basic",
                            fold_device=fold_device,
                            fold_batch_size=args.fold_batch_size,
                            fold_plddt_threshold=args.fold_plddt_threshold,
                            eval_samples_fold_max=args.eval_samples_fold_max,
                            fold_cache_dir=args.fold_cache_dir,
                        )
                        valids = _filter_valid_from_records(per_valid)
                        for s in valids:
                            if s not in seen:
                                seen.add(s)
                                acc.append(s)
                            if len(acc) >= n_valid_min:
                                break
                        rounds += 1
                    return acc

                seqs_only_b = _accumulate_valid_sequences(before_model_for_eval)
                seqs_only_a = _accumulate_valid_sequences(trainer.accelerator.unwrap_model(trainer.model))
                before_valid_count = len(seqs_only_b)
                after_valid_count = len(seqs_only_a)
                before_feasible = before_valid_count >= n_valid_min
                after_feasible = after_valid_count >= n_valid_min

        # Cap pairs to unique available per side to avoid replacement when filter_after budget is used
        def _unique_pairs_cap(nv: int) -> int:
            return int(max(0, nv * (nv - 1) // 2))

        pairs_used_before = int(args.pairwise_num_pairs)
        pairs_used_after = int(args.pairwise_num_pairs)

        if valid_filter != "none" and valid_strategy == "filter_after":
            pairs_used_before = min(pairs_used_before, _unique_pairs_cap(len(seqs_only_b)))
            pairs_used_after = min(pairs_used_after, _unique_pairs_cap(len(seqs_only_a)))

        def _compute_topk_for_sequences_with_pairs(seq_list, num_pairs_override: int):
            return topk_distance_avg(
                seq_list,
                mode=args.pairwise_distance_mode,
                topk_percent=args.pairwise_topk_percent,
                num_pairs=num_pairs_override,
                seed=args.pairwise_seed,
                gap_penalty=args.pairwise_gap_penalty,
            )

        before_topk_dist = _compute_topk_for_sequences_with_pairs(seqs_only_b, pairs_used_before)
        after_topk_dist = _compute_topk_for_sequences_with_pairs(seqs_only_a, pairs_used_after)

        report.update({
            "pairwise_distance_mode": args.pairwise_distance_mode,
            "pairwise_topk_percent": int(args.pairwise_topk_percent),
            "pairwise_num_pairs": int(args.pairwise_num_pairs),
            "pairwise_validity_filter": args.pairwise_validity_filter,
            "pairwise_valid_strategy": args.pairwise_valid_strategy,
            "before_topk_distance_avg": float(before_topk_dist),
            "after_topk_distance_avg": float(after_topk_dist),
        })
        if valid_filter != "none":
            report.update({
                "before_valid_pairs_target": int(args.pairwise_num_pairs),
                "after_valid_pairs_target": int(args.pairwise_num_pairs),
                "before_valid_count": int(before_valid_count or 0),
                "after_valid_count": int(after_valid_count or 0),
                "before_valid_pairs_feasible": bool(before_feasible),
                "after_valid_pairs_feasible": bool(after_feasible),
                "pairwise_collect_max_rounds": int(getattr(args, "pairwise_collect_max_rounds", 10)),
                "pairwise_eval_budget": int(getattr(args, "pairwise_eval_budget", 0) or 0),
                "before_pairs_used": int(pairs_used_before),
                "after_pairs_used": int(pairs_used_after),
            })
    except Exception:
        pass
    # Optional Vendi diversity computation (before/after)
    if args.compute_vendi:
        vendi_before = None
        vendi_after = None
        vendi_sigma_used_before = None
        vendi_sigma_used_after = None
        try:
            from utils.vendi_diversity import vendi_from_sequences  # lazy import
            vendi_device = args.vendi_device
            if vendi_device == "auto":
                vendi_device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
            # before
            seqs_only_b = [a for a, _ in seqs_before]
            weights_b = [p for _, p in seqs_before]
            res_b = vendi_from_sequences(
                seqs_only_b,
                weights=weights_b,
                model_name=args.vendi_model,
                device=vendi_device,
                dtype=args.vendi_dtype,
                batch_size=args.vendi_batch_size,
                kernel=args.vendi_kernel,
                sigma=args.vendi_sigma,
            )
            vendi_before = float(res_b.get("vendi_score", float("nan")))
            vendi_sigma_used_before = res_b.get("sigma_used")
            vendi_lambda1_over_trace_before = res_b.get("debug", {}).get("lambda1_over_trace")
            # after
            seqs_only_a = [a for a, _ in seqs_after]
            weights_a = [p for _, p in seqs_after]
            res_a = vendi_from_sequences(
                seqs_only_a,
                weights=weights_a,
                model_name=args.vendi_model,
                device=vendi_device,
                dtype=args.vendi_dtype,
                batch_size=args.vendi_batch_size,
                kernel=args.vendi_kernel,
                sigma=args.vendi_sigma,
            )
            vendi_after = float(res_a.get("vendi_score", float("nan")))
            vendi_sigma_used_after = res_a.get("sigma_used")
            vendi_lambda1_over_trace_after = res_a.get("debug", {}).get("lambda1_over_trace")
            report.update({
                "diversity_metric_used": "vendi",
                "before_diversity": vendi_before,
                "after_diversity": vendi_after,
                "before_vendi_score": vendi_before,
                "after_vendi_score": vendi_after,
                "before_vendi_sigma_used": vendi_sigma_used_before,
                "after_vendi_sigma_used": vendi_sigma_used_after,
                "before_vendi_lambda1_over_trace": vendi_lambda1_over_trace_before,
                "after_vendi_lambda1_over_trace": vendi_lambda1_over_trace_after,
                "vendi_model": args.vendi_model,
                "vendi_kernel": args.vendi_kernel,
                "vendi_dtype": args.vendi_dtype,
            })
        except Exception as e:
            # Best-effort debug log
            try:
                with open(os.path.join(args.out_dir, "vendi_debug.log"), "a") as lf:
                    lf.write(f"EXC {repr(e)}\n")
            except Exception:
                pass
    # When esmfold validity is active, summarize pLDDT
    if args.validity_mode == "esmfold":
        import statistics as stats
        def _mean_plddt_from_records(records: List[dict]):
            vals = [r.get("plddt_mean") for r in records if r.get("plddt_mean") is not None]
            return float(sum(vals) / len(vals)) if len(vals) > 0 else float("nan")
        # Reload the per-records from CSVs for simplicity
        before_valid_csv = os.path.join(args.out_dir, "before_validity.csv")
        after_valid_csv = os.path.join(args.out_dir, "after_validity.csv")
        def _load_plddt(path: str) -> List[dict]:
            out: List[dict] = []
            try:
                with open(path, "r") as f:
                    rr = csv.reader(f)
                    header = next(rr, None)
                    for row in rr:
                        if len(row) >= 6:
                            out.append({
                                "sequence": row[0],
                                "valid": int(row[1]),
                                "plddt_mean": None if row[2] == '' else float(row[2]),
                                "plddt_median": None if row[3] == '' else float(row[3]),
                                "fold_ok": row[4] == 'True',
                                "fold_error": row[5] if row[5] else None,
                            })
            except Exception:
                pass
            return out
        before_recs = _load_plddt(before_valid_csv)
        after_recs = _load_plddt(after_valid_csv)
        report.update({
            "before_mean_plddt": _mean_plddt_from_records(before_recs),
            "after_mean_plddt": _mean_plddt_from_records(after_recs),
        })
    # Log to WandB if enabled
    try:
        log_report(wandb_run, report)
    except Exception:
        pass

    # Write JSON unless disabled
    if not args.no_json_report:
        with open(os.path.join(args.out_dir, "grpo_exact_entropy.json"), "w") as f:
            json.dump(report, f, indent=2)
    # 'after_sequence_probs.csv' already written by dump_sequence_probs
    try:
        wandb_finish(wandb_run)
    except Exception:
        pass


if __name__ == "__main__":
    main()


