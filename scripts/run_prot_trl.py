import argparse
import os
import sys
import json
import math
import csv
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
    args = parser.parse_args()

    if GRPOTrainer is None or GRPOConfig is None:
        raise RuntimeError("trl[grpo] is required. Install a version that provides GRPOTrainer and GRPOConfig.")

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Prefer NVIDIA GPU if available, otherwise CPU
    device = get_preferred_device()
    try:
        torch.set_default_device(device)
    except Exception:
        pass

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
        ref_model_sync_steps=20,
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
    def dump_sequence_probs(model, csv_path, validity_csv_path):
        # Monte Carlo estimate of distribution and entropy without legality constraints
        H, H_valid, seqs, mean_valid, per_valid, mean_token_len, mean_residue_len = sample_entropy_and_validity(
            model,
            tok,
            args.horizon,
            max(args.eval_samples, args.batch_size),
            do_sample=args.eval_do_sample,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            temperature=args.eval_temperature,
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "probability"])  # empirical from samples
            for a, p in seqs:
                w.writerow([a, p])
        with open(validity_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "valid"])  # binary oracle
            for s, v in per_valid:
                # Ensure CSV reflects the 0/1 oracle already used during sampling
                w.writerow([s, int(v)])
        return H, H_valid, seqs, mean_valid, mean_token_len, mean_residue_len

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
    with open(os.path.join(args.out_dir, "grpo_exact_entropy.json"), "w") as f:
        json.dump(report, f, indent=2)
    # 'after_sequence_probs.csv' already written by dump_sequence_probs


if __name__ == "__main__":
    main()


