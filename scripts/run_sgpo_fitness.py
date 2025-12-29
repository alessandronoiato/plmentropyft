#!/usr/bin/env python3
"""
SGPO Fitness Oracle Integration Script

This script integrates the fitness oracle from the SGPO repository
(https://github.com/jsunn-y/SGPO) for protein fitness optimization.

Target dataset: TrpB (tryptophan synthase beta subunit)

The objective we optimize is the first variation of:
    J(π) = E_π[r(y)] + μ H(π) - η KL(π || π_base)

where r(y) is the fitness score from the SGPO oracle.

Workflow:
    1. Generate full sequences with ProGen2 (fine-tuned on TrpB MSA)
    2. Align to parent using MAFFT
    3. Project to 15 mutated positions
    4. Score with oracle ensemble

Usage:
    # Explore mode (analyze data, test components)
    python scripts/run_sgpo_fitness.py --mode explore --sgpo_repo ~/Code/SGPO

    # Training mode
    python scripts/run_sgpo_fitness.py --mode train --sgpo_repo ~/Code/SGPO \
        --steps 100 --batch_size 32 --fitness_scale 1.0

    # Evaluation mode
    python scripts/run_sgpo_fitness.py --mode eval --sgpo_repo ~/Code/SGPO \
        --eval_samples 500

    # Pareto sweep mode (for comparing with DPO)
    python scripts/run_sgpo_fitness.py --mode pareto --sgpo_repo ~/Code/SGPO \
        --pareto_entropy_coefs "0.01,0.05,0.1,0.2,0.5" --steps 100
"""

# Suppress warnings BEFORE any imports
import warnings
import os
import sys

# Suppress common HuggingFace/transformers warnings
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*model of type progen.*")
warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS.*")

# Also set logging level for transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import argparse
import json
import random

# Ensure remote code is trusted in non-interactive/batch environments
os.environ["HF_ALLOW_CODE_EXECUTION"] = "1"
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"

# Monkeypatch AutoConfig.from_pretrained to handle progen model type
# TRL calls AutoConfig internally with model paths that may not have custom code
from transformers import AutoConfig as _AutoConfig
from transformers import PretrainedConfig
_old_ac_from_pretrained = _AutoConfig.from_pretrained

def _ac_from_pretrained_trust(*args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    try:
        return _old_ac_from_pretrained(*args, **kwargs)
    except ValueError as e:
        if "progen" in str(e).lower():
            # Fall back to loading as generic config for progen models
            # This happens when TRL tries to reload from a local path without custom code
            config = PretrainedConfig.from_pretrained(*args, **kwargs)
            config.model_type = "progen"
            # Set architectures so TRL can find our registered class
            if not hasattr(config, 'architectures') or config.architectures is None:
                config.architectures = ["ProGenForCausalLM"]
            return config
        raise

_AutoConfig.from_pretrained = _ac_from_pretrained_trust

import numpy as np
import torch
from collections import Counter
from itertools import combinations

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import from utils.sgpo
from utils.sgpo import (
    MAFFTAligner,
    ProGen2Wrapper,
    SGPOFitnessOracle,
    SGPOPipeline,
    TrpBProjector,
    TRPB_COMBO_POSITIONS,
    TRPB_INFO,
    TRPB_PARENT_SEQUENCE,
    TRPB_WT_COMBO,
    analyze_fitness_distribution,
    combo_to_full_sequence,
    full_sequence_to_combo,
    load_trpb_fitness_data,
    make_fitness_reward,
)


# =============================================================================
# DIVERSITY METRICS (matching SGPO's analysis.ipynb)
# =============================================================================

def hamming_distance(seq1: str, seq2: str) -> int:
    """Compute Hamming distance between two sequences of equal length."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def pairwise_diversity(sequences: list) -> float:
    """Compute mean pairwise Hamming distance across all sequence pairs."""
    if len(sequences) < 2:
        return 0.0
    pairs = list(combinations(sequences, 2))
    if not pairs:
        return 0.0
    distances = [hamming_distance(a, b) for a, b in pairs]
    return float(np.mean(distances))


def shannon_entropy(sequences: list) -> float:
    """
    Compute Shannon entropy over a set of aligned sequences.
    Returns mean entropy across all positions.
    """
    if not sequences or len(sequences) == 0:
        return 0.0
    seq_len = len(sequences[0])
    if seq_len == 0:
        return 0.0
    
    entropy = []
    for pos in range(seq_len):
        column = [seq[pos] for seq in sequences if len(seq) > pos]
        if not column:
            continue
        counts = Counter(column)
        n = len(column)
        freqs = np.array([count / n for count in counts.values()])
        # Avoid log(0)
        freqs = freqs[freqs > 0]
        if len(freqs) > 0:
            entropy.append(-np.sum(freqs * np.log2(freqs)))
    
    return float(np.mean(entropy)) if entropy else 0.0


def compute_diversity_metrics(
    combos: list,
    parent_combo: str,
    full_sequences: list = None,
    parent_sequence: str = None,
) -> dict:
    """
    Compute all diversity metrics matching SGPO's analysis.
    
    Args:
        combos: List of 15-character combo sequences
        parent_combo: Parent/wild-type combo sequence
        full_sequences: Optional list of full sequences (for full-length metrics)
        parent_sequence: Optional parent full sequence
    
    Returns:
        Dictionary with diversity metrics
    """
    metrics = {}
    
    # Basic counts
    metrics["total_count"] = len(combos)
    metrics["unique_count"] = len(set(combos))
    metrics["uniqueness_ratio"] = metrics["unique_count"] / metrics["total_count"] if metrics["total_count"] > 0 else 0.0
    
    # Shannon entropy (combo level)
    metrics["shannon_entropy_combo"] = shannon_entropy(combos)
    
    # Pairwise diversity (combo level) - can be slow for large sets
    unique_combos = list(set(combos))
    if len(unique_combos) <= 1000:  # Limit for computational tractability
        metrics["pairwise_diversity_combo"] = pairwise_diversity(unique_combos)
    else:
        # Sample for large sets
        sampled = random.sample(unique_combos, 1000)
        metrics["pairwise_diversity_combo"] = pairwise_diversity(sampled)
        metrics["pairwise_diversity_sampled"] = True
    
    # Hamming distance to parent (combo level)
    hamming_to_parent = [hamming_distance(combo, parent_combo) for combo in combos]
    metrics["mean_hamming_to_parent"] = float(np.mean(hamming_to_parent))
    metrics["std_hamming_to_parent"] = float(np.std(hamming_to_parent))
    metrics["min_hamming_to_parent"] = int(np.min(hamming_to_parent))
    metrics["max_hamming_to_parent"] = int(np.max(hamming_to_parent))
    
    # Distribution of Hamming distances
    hamming_counts = Counter(hamming_to_parent)
    metrics["hamming_distribution"] = {str(k): v for k, v in sorted(hamming_counts.items())}
    
    # Full sequence metrics (if provided)
    if full_sequences and parent_sequence:
        metrics["shannon_entropy_full"] = shannon_entropy(full_sequences)
        hamming_full = [hamming_distance(seq, parent_sequence) for seq in full_sequences if len(seq) == len(parent_sequence)]
        if hamming_full:
            metrics["mean_hamming_to_parent_full"] = float(np.mean(hamming_full))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="SGPO Fitness Oracle Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="explore", 
                        choices=["explore", "train", "eval", "pareto"],
                        help="Mode: explore (analyze data), train (run GRPO), eval (evaluate), pareto (Pareto sweep)")
    
    # ===== SGPO Paths =====
    parser.add_argument("--sgpo_repo", type=str, default=None,
                        help="Path to cloned SGPO repository")
    parser.add_argument("--fitness_data", type=str, default=None,
                        help="Path to fitness CSV file (e.g., data/TrpB/fitness.csv)")
    parser.add_argument("--oracle_checkpoint", type=str, default=None,
                        help="Path to oracle checkpoints (e.g., oracle/checkpoints/TrpB/)")
    parser.add_argument("--progen2_path", type=str, default=None,
                        help="Path to fine-tuned ProGen2 model (e.g., checkpoints/causalLM_finetune/TrpB/best/)")
    parser.add_argument("--parent_fasta", type=str, default=None,
                        help="Path to parent.fasta file (e.g., data/TrpB/parent.fasta)")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="TrpB",
                        choices=["TrpB", "CreiLOV", "GB1"],
                        help="SGPO dataset to use")
    
    # ===== Model/Generation =====
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--generation_temp", type=float, default=1.0,
                        help="Generation temperature")
    parser.add_argument("--generation_top_p", type=float, default=0.95,
                        help="Nucleus sampling threshold")
    parser.add_argument("--generation_batch_size", type=int, default=40,
                        help="Generation batch size")
    
    # ===== MAFFT =====
    parser.add_argument("--mafft_path", type=str, default="mafft",
                        help="Path to MAFFT binary")
    
    # ===== Oracle =====
    parser.add_argument("--impose_penalty", action="store_true",
                        help="Apply Hamming distance penalty to fitness")
    parser.add_argument("--penalty_cutoff", type=int, default=233,
                        help="Hamming distance threshold for penalty")
    parser.add_argument("--penalty_rate", type=float, default=0.99,
                        help="Exponential decay rate for penalty")
    
    # ===== Reward Coefficients =====
    parser.add_argument("--fitness_scale", type=float, default=1.0,
                        help="Scale factor λ for fitness reward")
    parser.add_argument("--entropy_coef", type=float, default=0.1,
                        help="Entropy coefficient μ")
    parser.add_argument("--first_variation_coef", type=float, default=0.0,
                        help="KL-to-base coefficient η")
    
    # ===== Training =====
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num_generations", type=int, default=16,
                        help="Number of generations per prompt")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient (β)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # ===== Evaluation =====
    parser.add_argument("--eval_samples", type=int, default=1000,
                        help="Number of samples for evaluation")
    
    # ===== Pareto Sweep =====
    parser.add_argument("--pareto_fitness_scales", type=str, default="1.0",
                        help="Comma-separated fitness_scale values for Pareto sweep (keep fixed)")
    parser.add_argument("--pareto_entropy_coefs", type=str, default="0.01,0.05,0.1,0.2,0.5",
                        help="Comma-separated entropy_coef (μ) values for Pareto sweep")
    parser.add_argument("--pareto_repeats", type=int, default=3,
                        help="Number of repeats per Pareto configuration")
    
    # ===== Output =====
    parser.add_argument("--out_dir", type=str, default=os.path.join(_PROJECT_ROOT, "outputs/sgpo"),
                        help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="sgpo-fitness",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose/debug output (progress bars, debug prints)")
    
    args = parser.parse_args()
    
    # Set verbose flag globally for SGPO modules
    os.environ["SGPO_VERBOSE"] = "1" if args.verbose else "0"
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 60)
    print("SGPO Fitness Oracle Integration")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print()
    
    # =========================================================================
    # EXPLORE MODE
    # =========================================================================
    if args.mode == "explore":
        print(TRPB_INFO)
        
        # Infer paths from sgpo_repo if provided
        fitness_data = args.fitness_data
        oracle_ckpt = args.oracle_checkpoint
        progen2_path = args.progen2_path
        parent_fasta = args.parent_fasta
        
        if args.sgpo_repo and os.path.isdir(args.sgpo_repo):
            if not fitness_data:
                fitness_data = os.path.join(args.sgpo_repo, "data", args.dataset, "fitness.csv")
            if not oracle_ckpt:
                oracle_ckpt = os.path.join(args.sgpo_repo, "oracle", "checkpoints", args.dataset)
            if not progen2_path:
                progen2_path = os.path.join(args.sgpo_repo, "checkpoints", "causalLM_finetune", args.dataset, "best")
            if not parent_fasta:
                parent_fasta = os.path.join(args.sgpo_repo, "data", args.dataset, "parent.fasta")
        
        # Load and analyze fitness data
        if fitness_data and os.path.exists(fitness_data):
            sequences, fitness_values, records = load_trpb_fitness_data(fitness_data)
            stats = analyze_fitness_distribution(fitness_values, f"{args.dataset} fitness")
            
            print("\n[Split distribution]")
            split_counts = {}
            for r in records:
                s = r.get("split", "unknown")
                split_counts[s] = split_counts.get(s, 0) + 1
            for s, c in sorted(split_counts.items()):
                print(f"  {s}: {c}")
            
            print("\n[Mutation count distribution]")
            nmut_counts = {}
            for r in records:
                n = r.get("n_mut", 0)
                nmut_counts[n] = nmut_counts.get(n, 0) + 1
            for n, c in sorted(nmut_counts.items()):
                print(f"  n_mut={n}: {c}")
            
            print("\n[Top 5 fitness sequences]")
            sorted_records = sorted(records, key=lambda x: x["fitness"], reverse=True)[:5]
            for r in sorted_records:
                print(f"  {r['combo']} | {r['mut']:<30} | fitness={r['fitness']:.4f}")
            
            # Test sequence conversion
            print("\n[Testing sequence conversion]")
            wt_combo = TRPB_WT_COMBO
            full_seq = combo_to_full_sequence(wt_combo)
            combo_back = full_sequence_to_combo(full_seq)
            print(f"  WT Combo: {wt_combo}")
            print(f"  Full seq length: {len(full_seq)}")
            print(f"  Round-trip match: {combo_back == wt_combo}")
            
            # Save stats
            stats["split_distribution"] = split_counts
            stats["nmut_distribution"] = nmut_counts
            stats_path = os.path.join(args.out_dir, f"{args.dataset}_fitness_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n[Saved stats to {stats_path}]")
        else:
            print("\nTo analyze fitness data, provide --fitness_data or --sgpo_repo path.")
            print("\nExpected SGPO data structure:")
            print("  {sgpo_repo}/data/TrpB/fitness.csv")
            print("  {sgpo_repo}/oracle/checkpoints/TrpB/")
            print("  {sgpo_repo}/checkpoints/causalLM_finetune/TrpB/best/")
            print("  {sgpo_repo}/data/TrpB/parent.fasta")
        
        # Test oracle wrapper
        print("\n[Testing Oracle Wrapper]")
        oracle = SGPOFitnessOracle(
            checkpoint_dir=oracle_ckpt or "",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dataset=args.dataset,
            parent_combo=TRPB_WT_COMBO,
            impose_penalty=args.impose_penalty,
        )
        test_seqs = [TRPB_WT_COMBO, "AAALIYVFGSVSGSY", "TAALIYVLGSKGGSY"]
        test_scores = oracle(test_seqs)
        print(f"  Test combo sequences: {test_seqs}")
        print(f"  Test scores: {test_scores}")
        
        # Test ProGen2 if available
        if progen2_path and os.path.isdir(progen2_path):
            print("\n[Testing ProGen2 Wrapper]")
            progen2 = ProGen2Wrapper(
                model_path=progen2_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                sgpo_repo=args.sgpo_repo,
            )
            if progen2.model is not None:
                print("  Generating 3 test sequences...")
                test_generated = progen2.sample(num_return_sequences=3, temperature=1.0, top_p=0.95)
                for i, seq in enumerate(test_generated):
                    print(f"    [{i}] len={len(seq)}: {seq[:50]}...")
        
        # Test MAFFT
        print("\n[Testing MAFFT Aligner]")
        aligner = MAFFTAligner(
            parent_sequence=TRPB_PARENT_SEQUENCE,
            parent_fasta_path=parent_fasta if parent_fasta and os.path.exists(parent_fasta) else None,
            mafft_path=args.mafft_path,
        )
        if aligner.is_available:
            print("  MAFFT is available")
        else:
            print("  MAFFT not found - will use fallback alignment")
        
    # =========================================================================
    # TRAIN MODE
    # =========================================================================
    elif args.mode == "train":
        print("[Train mode] Setting up GRPO training with fitness reward...")
        
        # Infer paths
        fitness_data = args.fitness_data
        oracle_ckpt = args.oracle_checkpoint
        progen2_path = args.progen2_path
        parent_fasta = args.parent_fasta
        
        if args.sgpo_repo and os.path.isdir(args.sgpo_repo):
            if not fitness_data:
                fitness_data = os.path.join(args.sgpo_repo, "data", args.dataset, "fitness.csv")
            if not oracle_ckpt:
                oracle_ckpt = os.path.join(args.sgpo_repo, "oracle", "checkpoints", args.dataset)
            if not progen2_path:
                progen2_path = os.path.join(args.sgpo_repo, "checkpoints", "causalLM_finetune", args.dataset, "best")
            if not parent_fasta:
                parent_fasta = os.path.join(args.sgpo_repo, "data", args.dataset, "parent.fasta")
        
        # Validate required paths
        if not progen2_path or not os.path.isdir(progen2_path):
            print(f"[ERROR] ProGen2 model path not found: {progen2_path}")
            print("  Provide --progen2_path or --sgpo_repo with downloaded checkpoints")
            sys.exit(1)
        
        if not oracle_ckpt or not os.path.isdir(oracle_ckpt):
            print(f"[ERROR] Oracle checkpoint path not found: {oracle_ckpt}")
            print("  Provide --oracle_checkpoint or --sgpo_repo with downloaded checkpoints")
            sys.exit(1)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
        print(f"  ProGen2 path: {progen2_path}")
        print(f"  Oracle path: {oracle_ckpt}")
        
        # Initialize components
        print("\n[1/4] Loading ProGen2 model...")
        progen2 = ProGen2Wrapper(
            model_path=progen2_path,
            device=device,
            sgpo_repo=args.sgpo_repo,
        )
        
        if progen2.model is None:
            print("[ERROR] Failed to load ProGen2 model")
            sys.exit(1)
        
        print("\n[2/4] Setting up MAFFT aligner...")
        aligner = MAFFTAligner(
            parent_sequence=TRPB_PARENT_SEQUENCE,
            parent_fasta_path=parent_fasta if parent_fasta and os.path.exists(parent_fasta) else None,
            mafft_path=args.mafft_path,
        )
        
        print("\n[3/4] Setting up projector...")
        projector = TrpBProjector(
            parent_sequence=TRPB_PARENT_SEQUENCE,
            mutated_positions=TRPB_COMBO_POSITIONS,
        )
        
        print("\n[4/4] Loading oracle ensemble...")
        oracle = SGPOFitnessOracle(
            checkpoint_dir=oracle_ckpt,
            device=device,
            dataset=args.dataset,
            parent_combo=TRPB_WT_COMBO,
            impose_penalty=args.impose_penalty,
            penalty_cutoff=args.penalty_cutoff,
            penalty_rate=args.penalty_rate,
        )
        
        # Create pipeline
        pipeline = SGPOPipeline(
            model=progen2,
            aligner=aligner,
            projector=projector,
            oracle=oracle,
        )
        
        # Setup WandB
        if not args.no_wandb:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or f"sgpo-{args.dataset}-{args.seed}",
                    config=vars(args),
                )
                print("\n[WandB] Initialized")
            except Exception as e:
                print(f"[WandB] Warning: {e}")
        
        # Reward function will be created after trainer (needs ref_model access)
        print("\n[Reward function coefficients]")
        print(f"  Fitness scale (λ): {args.fitness_scale}")
        print(f"  Entropy coef (μ): {args.entropy_coef}")
        print(f"  First variation coef (η): {args.first_variation_coef}")
        
        # GRPO Training Setup
        print("\n[Setting up GRPO Trainer]")
        try:
            from trl import GRPOConfig, GRPOTrainer
            
            # Training config - use bf16 only on CUDA
            use_bf16 = torch.cuda.is_available()
            grpo_config = GRPOConfig(
                output_dir=os.path.join(args.out_dir, "trainer_output"),
                num_train_epochs=1,
                max_steps=args.steps,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                beta=args.beta,
                num_generations=args.num_generations,
                logging_steps=10,
                save_steps=max(args.steps // 5, 1),
                report_to="wandb" if not args.no_wandb else "none",
                bf16=use_bf16,
                fp16=False,
                gradient_checkpointing=False,  # ProGen doesn't support this
                seed=args.seed,  # Use specified seed for reproducibility
                disable_tqdm=not args.verbose,  # Suppress progress bars unless verbose
            )
            
            # Create dataset of prompts (GRPO uses reward function, not labeled data)
            # ProGen2 uses "1" as the start/context token (token_id=3)
            from datasets import Dataset
            prompts = ["1"] * args.batch_size
            train_ds = Dataset.from_dict({"prompt": prompts})
            
            print(f"  Steps: {args.steps}")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Generations per prompt: {args.num_generations}")
            print(f"  Learning rate: {args.learning_rate}")
            print(f"  Beta (KL): {args.beta}")
            
            # Register custom model class on transformers namespace for TRL compatibility
            # Also fix model config._name_or_path to point to HF repo with custom code
            try:
                import transformers
                import transformers.utils.import_utils as _tf_import_utils
                _model_cls = progen2.model.__class__
                _model_cls_name = _model_cls.__name__
                
                # Store custom classes for the patched __getattr__
                _custom_classes = {
                    _model_cls_name: _model_cls,
                    "ProGenForCausalLM": _model_cls,
                }
                
                # Monkey-patch the transformers module's __getattr__ to return our custom classes
                _original_getattr = _tf_import_utils._LazyModule.__getattr__
                def _patched_getattr(self, name):
                    if name in _custom_classes:
                        return _custom_classes[name]
                    return _original_getattr(self, name)
                _tf_import_utils._LazyModule.__getattr__ = _patched_getattr
                
                print(f"  Patched transformers lazy loader for {_model_cls_name} and ProGenForCausalLM")
                
                # Keep the original model path - don't redirect to HF repo
                # The patched lazy loader handles the class lookup
                if hasattr(progen2.model, 'config') and hasattr(progen2.model.config, '_name_or_path'):
                    print(f"  Model config path: {progen2.model.config._name_or_path}")
            except Exception as e:
                import traceback
                print(f"  Warning: could not register model class: {e}")
                traceback.print_exc()
            
            # Create a mutable wrapper that will hold the real reward function
            # This allows us to pass it to trainer, then update it after trainer is created
            class RewardWrapper:
                __name__ = "fitness_reward"  # TRL expects this attribute
                
                def __init__(self):
                    self.real_reward_func = None
                    self._fitness_log = []
                
                def __call__(self, *args, **kwargs):
                    if self.real_reward_func is not None:
                        return self.real_reward_func(*args, **kwargs)
                    # Fallback if called before setup (shouldn't happen)
                    return [0.0] * len(kwargs.get('prompts', args[0] if args else []))
                
                def get_fitness_log(self):
                    if self.real_reward_func is not None and hasattr(self.real_reward_func, 'get_fitness_log'):
                        return self.real_reward_func.get_fitness_log()
                    return self._fitness_log
            
            reward_wrapper = RewardWrapper()
            
            # Create trainer with wrapper reward
            trainer = GRPOTrainer(
                model=progen2.model,
                args=grpo_config,
                reward_funcs=[reward_wrapper],
                train_dataset=train_ds,
                eval_dataset=train_ds,
                processing_class=progen2.tokenizer,
            )
            
            # Now create real reward function with access to trainer's ref_model
            print("\n[Setting up reward function with trainer access]")
            reward_func = make_fitness_reward(
                pipeline=pipeline,
                trainer=trainer,
                tokenizer=progen2.tokenizer,
                entropy_coef=args.entropy_coef,
                first_variation_coef=args.first_variation_coef,
                fitness_scale=args.fitness_scale,
                base_ref_model=None,  # Could pass SGPO's base model if needed
                out_dir=args.out_dir,
            )
            
            # Update wrapper to use real function
            reward_wrapper.real_reward_func = reward_func
            
            print("\n[Starting training...]")
            trainer.train()
            
            # Save final model
            final_path = os.path.join(args.out_dir, "final_model")
            trainer.save_model(final_path)
            print(f"\n[Saved final model to {final_path}]")
            
            # Save fitness log
            fitness_log = reward_func.get_fitness_log()
            log_path = os.path.join(args.out_dir, "fitness_log.json")
            with open(log_path, "w") as f:
                json.dump(fitness_log, f, indent=2)
            print(f"[Saved fitness log to {log_path}]")
            
        except ImportError as e:
            print(f"[ERROR] TRL not installed: {e}")
            print("  Install with: pip install trl")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # =========================================================================
    # EVAL MODE
    # =========================================================================
    elif args.mode == "eval":
        print("[Eval mode] Evaluating sequences with fitness oracle...")
        
        # Infer paths
        oracle_ckpt = args.oracle_checkpoint
        progen2_path = args.progen2_path
        parent_fasta = args.parent_fasta
        
        if args.sgpo_repo and os.path.isdir(args.sgpo_repo):
            if not oracle_ckpt:
                oracle_ckpt = os.path.join(args.sgpo_repo, "oracle", "checkpoints", args.dataset)
            if not progen2_path:
                progen2_path = os.path.join(args.sgpo_repo, "checkpoints", "causalLM_finetune", args.dataset, "best")
            if not parent_fasta:
                parent_fasta = os.path.join(args.sgpo_repo, "data", args.dataset, "parent.fasta")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load components
        print("\n[Loading components...]")
        progen2 = ProGen2Wrapper(
            model_path=progen2_path,
            device=device,
            sgpo_repo=args.sgpo_repo,
        )
        
        aligner = MAFFTAligner(
            parent_sequence=TRPB_PARENT_SEQUENCE,
            parent_fasta_path=parent_fasta if parent_fasta and os.path.exists(parent_fasta) else None,
            mafft_path=args.mafft_path,
        )
        
        projector = TrpBProjector(
            parent_sequence=TRPB_PARENT_SEQUENCE,
            mutated_positions=TRPB_COMBO_POSITIONS,
        )
        
        oracle = SGPOFitnessOracle(
            checkpoint_dir=oracle_ckpt or "",
            device=device,
            dataset=args.dataset,
            parent_combo=TRPB_WT_COMBO,
            impose_penalty=args.impose_penalty,
        )
        
        # Create pipeline
        pipeline = SGPOPipeline(
            model=progen2,
            aligner=aligner,
            projector=projector,
            oracle=oracle,
        )
        
        # Generate and evaluate
        print(f"\n[Generating {args.eval_samples} sequences...]")
        results = pipeline.generate_and_score(
            num_sequences=args.eval_samples,
            temperature=args.generation_temp,
            top_p=args.generation_top_p,
            batch_size=args.generation_batch_size,
        )
        
        fitness_scores = results["fitness_scores"]
        combos = results["combos"]
        projected_seqs = results.get("projected_sequences", [])
        
        # Statistics
        if fitness_scores:
            stats = analyze_fitness_distribution(fitness_scores, "Generated fitness")
            
            # Compute diversity metrics (matching SGPO's analysis)
            print("\n[Computing diversity metrics...]")
            diversity_metrics = compute_diversity_metrics(
                combos=combos,
                parent_combo=TRPB_WT_COMBO,
                full_sequences=projected_seqs if projected_seqs else None,
                parent_sequence=TRPB_PARENT_SEQUENCE,
            )
            
            # Print diversity summary
            print("\n" + "=" * 50)
            print("DIVERSITY METRICS (matching SGPO)")
            print("=" * 50)
            print(f"  Total sequences:      {diversity_metrics['total_count']}")
            print(f"  Unique sequences:     {diversity_metrics['unique_count']}")
            print(f"  Uniqueness ratio:     {diversity_metrics['uniqueness_ratio']:.2%}")
            print(f"  Shannon entropy:      {diversity_metrics['shannon_entropy_combo']:.4f}")
            print(f"  Pairwise diversity:   {diversity_metrics['pairwise_diversity_combo']:.4f}")
            print(f"  Mean Hamming to WT:   {diversity_metrics['mean_hamming_to_parent']:.2f} ± {diversity_metrics['std_hamming_to_parent']:.2f}")
            print(f"  Hamming range:        [{diversity_metrics['min_hamming_to_parent']}, {diversity_metrics['max_hamming_to_parent']}]")
            print("=" * 50)
            
            # Find best sequences
            print("\n[Top 10 generated sequences]")
            sorted_idx = np.argsort(fitness_scores)[::-1][:10]
            for i, idx in enumerate(sorted_idx):
                hamming = hamming_distance(combos[idx], TRPB_WT_COMBO)
                print(f"  [{i+1}] {combos[idx]} | fitness={fitness_scores[idx]:.4f} | Δ={hamming}")
            
            # Save results
            results_path = os.path.join(args.out_dir, "eval_results.json")
            with open(results_path, "w") as f:
                json.dump({
                    "fitness_stats": stats,
                    "diversity_metrics": diversity_metrics,
                    "n_samples": len(fitness_scores),
                    "combos": combos,
                    "fitness_scores": fitness_scores,
                }, f, indent=2)
            print(f"\n[Saved results to {results_path}]")
            
            # Save FASTA
            fasta_path = os.path.join(args.out_dir, "generated.fasta")
            with open(fasta_path, "w") as f:
                for i, (seq, fit) in enumerate(zip(projected_seqs if projected_seqs else combos, fitness_scores)):
                    f.write(f">{i}|fitness={fit:.4f}\n{seq}\n")
            print(f"[Saved sequences to {fasta_path}]")
        else:
            print("[WARNING] No sequences generated")
    
    # =========================================================================
    # PARETO MODE - Sweep over hyperparameters for Pareto frontier
    # =========================================================================
    elif args.mode == "pareto":
        print("[Pareto mode] Running Pareto sweep over fitness_scale and entropy_coef...")
        
        # Parse sweep parameters
        fitness_scales = [float(x) for x in args.pareto_fitness_scales.split(",")]
        entropy_coefs = [float(x) for x in args.pareto_entropy_coefs.split(",")]
        
        print(f"\n[Pareto sweep configuration]")
        print(f"  Fitness scales: {fitness_scales}")
        print(f"  Entropy coefs:  {entropy_coefs}")
        print(f"  Repeats:        {args.pareto_repeats}")
        print(f"  Steps per run:  {args.steps}")
        
        # Create sweep combinations
        sweep_configs = []
        for fs in fitness_scales:
            for ec in entropy_coefs:
                for rep in range(args.pareto_repeats):
                    sweep_configs.append({
                        "fitness_scale": fs,
                        "entropy_coef": ec,
                        "repeat": rep,
                        "seed": args.seed + rep,
                    })
        
        print(f"\n[Total configurations: {len(sweep_configs)}]")
        
        # Infer paths
        oracle_ckpt = args.oracle_checkpoint
        progen2_path = args.progen2_path
        parent_fasta = args.parent_fasta
        
        if args.sgpo_repo and os.path.isdir(args.sgpo_repo):
            if not oracle_ckpt:
                oracle_ckpt = os.path.join(args.sgpo_repo, "oracle", "checkpoints", args.dataset)
            if not progen2_path:
                progen2_path = os.path.join(args.sgpo_repo, "checkpoints", "causalLM_finetune", args.dataset, "best")
            if not parent_fasta:
                parent_fasta = os.path.join(args.sgpo_repo, "data", args.dataset, "parent.fasta")
        
        # Validate paths
        if not progen2_path or not os.path.isdir(progen2_path):
            print(f"[ERROR] ProGen2 model not found: {progen2_path}")
            sys.exit(1)
        if not oracle_ckpt or not os.path.isdir(oracle_ckpt):
            print(f"[ERROR] Oracle checkpoint not found: {oracle_ckpt}")
            sys.exit(1)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pareto_results = []
        
        # Run each configuration
        for cfg_idx, cfg in enumerate(sweep_configs):
            print(f"\n{'='*60}")
            print(f"[Config {cfg_idx+1}/{len(sweep_configs)}]")
            print(f"  fitness_scale={cfg['fitness_scale']}, entropy_coef={cfg['entropy_coef']}, repeat={cfg['repeat']}")
            print("=" * 60)
            
            # Set seed for this run
            run_seed = cfg["seed"]
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
            
            # Create output directory for this config
            cfg_name = f"fs{cfg['fitness_scale']}_ec{cfg['entropy_coef']}_rep{cfg['repeat']}"
            cfg_out_dir = os.path.join(args.out_dir, "pareto", cfg_name)
            os.makedirs(cfg_out_dir, exist_ok=True)
            
            try:
                # Initialize components (fresh for each run)
                print("\n[Loading model and components...]")
                progen2 = ProGen2Wrapper(
                    model_path=progen2_path,
                    device=device,
                    sgpo_repo=args.sgpo_repo,
                )
                
                aligner = MAFFTAligner(
                    parent_sequence=TRPB_PARENT_SEQUENCE,
                    parent_fasta_path=parent_fasta if parent_fasta and os.path.exists(parent_fasta) else None,
                    mafft_path=args.mafft_path,
                )
                
                projector = TrpBProjector(
                    parent_sequence=TRPB_PARENT_SEQUENCE,
                    mutated_positions=TRPB_COMBO_POSITIONS,
                )
                
                oracle = SGPOFitnessOracle(
                    checkpoint_dir=oracle_ckpt,
                    device=device,
                    dataset=args.dataset,
                    parent_combo=TRPB_WT_COMBO,
                    impose_penalty=args.impose_penalty,
                )
                
                pipeline = SGPOPipeline(
                    model=progen2,
                    aligner=aligner,
                    projector=projector,
                    oracle=oracle,
                )
                
                # GRPO Training
                from trl import GRPOConfig, GRPOTrainer
                from datasets import Dataset
                
                use_bf16 = torch.cuda.is_available()
                grpo_config = GRPOConfig(
                    output_dir=os.path.join(cfg_out_dir, "trainer_output"),
                    num_train_epochs=1,
                    max_steps=args.steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    learning_rate=args.learning_rate,
                    beta=args.beta,
                    num_generations=args.num_generations,
                    logging_steps=10,
                    save_strategy="no",  # Don't save checkpoints (ProGen config issues)
                    report_to="none",  # Disable wandb for sweep
                    bf16=use_bf16,
                    fp16=False,
                    gradient_checkpointing=False,  # ProGen doesn't support this
                    seed=run_seed,  # Ensure different seeds produce different results
                    disable_tqdm=not args.verbose,  # Suppress progress bars unless verbose
                )
                
                prompts = ["1"] * args.batch_size
                train_ds = Dataset.from_dict({"prompt": prompts})
                
                # Register model class for TRL
                import transformers
                import transformers.utils.import_utils as _tf_import_utils
                _model_cls = progen2.model.__class__
                _model_cls_name = _model_cls.__name__
                _custom_classes = {_model_cls_name: _model_cls, "ProGenForCausalLM": _model_cls}
                _original_getattr = _tf_import_utils._LazyModule.__getattr__
                def _patched_getattr(self, name):
                    if name in _custom_classes:
                        return _custom_classes[name]
                    return _original_getattr(self, name)
                _tf_import_utils._LazyModule.__getattr__ = _patched_getattr
                
                # Create a mutable wrapper that will hold the real reward function
                # This allows us to pass it to trainer, then update it after trainer is created
                class RewardWrapper:
                    __name__ = "fitness_reward"  # TRL expects this attribute
                    
                    def __init__(self):
                        self.real_reward_func = None
                        self._fitness_log = []
                    
                    def __call__(self, *args, **kwargs):
                        if self.real_reward_func is not None:
                            return self.real_reward_func(*args, **kwargs)
                        # Fallback if called before setup (shouldn't happen)
                        return [0.0] * len(kwargs.get('prompts', args[0] if args else []))
                    
                    def get_fitness_log(self):
                        if self.real_reward_func is not None and hasattr(self.real_reward_func, 'get_fitness_log'):
                            return self.real_reward_func.get_fitness_log()
                        return self._fitness_log
                
                reward_wrapper = RewardWrapper()
                
                trainer = GRPOTrainer(
                    model=progen2.model,
                    args=grpo_config,
                    reward_funcs=[reward_wrapper],
                    train_dataset=train_ds,
                    eval_dataset=train_ds,
                    processing_class=progen2.tokenizer,
                )
                
                # Now create the real reward function with trainer access
                reward_func = make_fitness_reward(
                    pipeline=pipeline,
                    trainer=trainer,
                    tokenizer=progen2.tokenizer,
                    entropy_coef=cfg["entropy_coef"],
                    first_variation_coef=args.first_variation_coef,
                    fitness_scale=cfg["fitness_scale"],
                    base_ref_model=None,
                    out_dir=cfg_out_dir,
                )
                # Update the wrapper to use the real function
                reward_wrapper.real_reward_func = reward_func
                
                print(f"\n[Training with fitness_scale={cfg['fitness_scale']}, entropy_coef={cfg['entropy_coef']}...]")
                trainer.train()
                
                # Evaluate after training
                print("\n[Evaluating trained model...]")
                results = pipeline.generate_and_score(
                    num_sequences=args.eval_samples,
                    temperature=args.generation_temp,
                    top_p=args.generation_top_p,
                    batch_size=args.generation_batch_size,
                )
                
                fitness_scores = results["fitness_scores"]
                combos = results["combos"]
                
                if fitness_scores:
                    stats = analyze_fitness_distribution(fitness_scores, f"Config {cfg_name}")
                    diversity = compute_diversity_metrics(combos, TRPB_WT_COMBO)
                    
                    result_entry = {
                        "fitness_scale": cfg["fitness_scale"],
                        "entropy_coef": cfg["entropy_coef"],
                        "repeat": cfg["repeat"],
                        "seed": cfg["seed"],
                        "mean_fitness": stats.get("mean", 0.0),
                        "std_fitness": stats.get("std", 0.0),
                        "max_fitness": stats.get("max", 0.0),
                        "q90_fitness": stats.get("q90", 0.0),
                        "unique_count": diversity["unique_count"],
                        "shannon_entropy": diversity["shannon_entropy_combo"],
                        "pairwise_diversity": diversity["pairwise_diversity_combo"],
                    }
                    pareto_results.append(result_entry)
                    
                    # Save per-config results
                    with open(os.path.join(cfg_out_dir, "results.json"), "w") as f:
                        json.dump(result_entry, f, indent=2)
                
                # Clean up
                del trainer, progen2, pipeline
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"[ERROR] Config {cfg_name} failed: {e}")
                import traceback
                print("\n" + "="*60)
                print("FULL TRACEBACK:")
                print("="*60)
                traceback.print_exc()
                print("="*60 + "\n")
                continue
        
        # Save Pareto summary
        if pareto_results:
            import pandas as pd
            pareto_df = pd.DataFrame(pareto_results)
            pareto_path = os.path.join(args.out_dir, "pareto", "pareto_summary.csv")
            pareto_df.to_csv(pareto_path, index=False)
            print(f"\n[Saved Pareto summary to {pareto_path}]")
            
            # Print summary
            print("\n" + "=" * 70)
            print("PARETO SWEEP SUMMARY")
            print("=" * 70)
            print(pareto_df.to_string(index=False))
            print("=" * 70)
    
    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
