#!/usr/bin/env python3
"""
Compare Pareto Frontiers: SGPO DPO vs Our GRPO

This script loads results from SGPO's DPO experiments and our GRPO experiments,
and plots them together to compare Pareto frontiers.

Usage:
    python scripts/compare_pareto.py \
        --sgpo_results ~/Code/SGPO/exps/protein/TrpB/causalLM_finetune/pareto/DPO/pareto_summary.csv \
        --grpo_results outputs/sgpo/pareto/pareto_summary.csv \
        --output outputs/sgpo/pareto_comparison.png
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_sgpo_pareto(csv_path: str) -> pd.DataFrame:
    """Load SGPO's pareto_summary.csv format."""
    df = pd.read_csv(csv_path)
    
    # SGPO columns: sequence, round, fitness, guidance_param, repeat, task
    # We need to aggregate by guidance_param
    
    # Filter to round=1 (after training, not baseline)
    if "round" in df.columns:
        df_trained = df[df["round"] == 1].copy()
    else:
        df_trained = df.copy()
    
    # Group by guidance_param and compute stats
    results = []
    for gp, group in df_trained.groupby("guidance_param"):
        fitness = group["fitness"].values
        sequences = group["sequence"].values if "sequence" in group.columns else []
        
        # Compute diversity (unique count and Shannon entropy proxy)
        unique_seqs = len(set(sequences)) if len(sequences) > 0 else 0
        uniqueness_ratio = unique_seqs / len(sequences) if len(sequences) > 0 else 0
        
        results.append({
            "method": "DPO",
            "guidance_param": gp,
            "mean_fitness": np.mean(fitness),
            "std_fitness": np.std(fitness),
            "max_fitness": np.max(fitness),
            "q90_fitness": np.percentile(fitness, 90),
            "n_samples": len(fitness),
            "unique_count": unique_seqs,
            "uniqueness_ratio": uniqueness_ratio,
        })
    
    return pd.DataFrame(results)


def load_grpo_pareto(csv_path: str) -> pd.DataFrame:
    """Load our GRPO pareto_summary.csv format."""
    df = pd.read_csv(csv_path)
    df["method"] = "GRPO"
    return df


def plot_pareto(
    dpo_df: pd.DataFrame,
    grpo_df: pd.DataFrame,
    output_path: str,
    x_metric: str = "unique_count",
    y_metric: str = "mean_fitness",
):
    """Plot Pareto frontiers comparing DPO and GRPO."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fitness vs Diversity (unique count)
    ax1 = axes[0]
    
    if len(dpo_df) > 0:
        ax1.scatter(
            dpo_df[x_metric], dpo_df[y_metric],
            c="blue", marker="o", s=100, alpha=0.7, label="DPO (SGPO)"
        )
        # Connect points to show frontier
        dpo_sorted = dpo_df.sort_values(x_metric)
        ax1.plot(dpo_sorted[x_metric], dpo_sorted[y_metric], "b--", alpha=0.5)
        
        # Annotate with guidance param
        for _, row in dpo_df.iterrows():
            ax1.annotate(
                f"β={row['guidance_param']:.2f}",
                (row[x_metric], row[y_metric]),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
                color="blue"
            )
    
    if len(grpo_df) > 0:
        # Aggregate GRPO by (fitness_scale, entropy_coef) across repeats
        grpo_agg = grpo_df.groupby(["fitness_scale", "entropy_coef"]).agg({
            x_metric: "mean",
            y_metric: "mean",
            "std_fitness": "mean",
        }).reset_index()
        
        ax1.scatter(
            grpo_agg[x_metric], grpo_agg[y_metric],
            c="red", marker="s", s=100, alpha=0.7, label="GRPO (Ours)"
        )
        # Connect points
        grpo_sorted = grpo_agg.sort_values(x_metric)
        ax1.plot(grpo_sorted[x_metric], grpo_sorted[y_metric], "r--", alpha=0.5)
        
        # Annotate with params
        for _, row in grpo_agg.iterrows():
            ax1.annotate(
                f"λ={row['fitness_scale']:.0f}",
                (row[x_metric], row[y_metric]),
                textcoords="offset points", xytext=(5, -10), fontsize=8,
                color="red"
            )
    
    ax1.set_xlabel("Diversity (Unique Sequences)", fontsize=12)
    ax1.set_ylabel("Mean Fitness", fontsize=12)
    ax1.set_title("Pareto Frontier: Fitness vs Diversity", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max/Q90 Fitness vs Diversity
    ax2 = axes[1]
    y_metric2 = "q90_fitness" if "q90_fitness" in grpo_df.columns else "max_fitness"
    
    if len(dpo_df) > 0:
        ax2.scatter(
            dpo_df[x_metric], dpo_df[y_metric2] if y_metric2 in dpo_df.columns else dpo_df["max_fitness"],
            c="blue", marker="o", s=100, alpha=0.7, label="DPO (SGPO)"
        )
    
    if len(grpo_df) > 0:
        grpo_agg = grpo_df.groupby(["fitness_scale", "entropy_coef"]).agg({
            x_metric: "mean",
            y_metric2: "mean" if y_metric2 in grpo_df.columns else lambda x: 0,
        }).reset_index()
        
        ax2.scatter(
            grpo_agg[x_metric], grpo_agg[y_metric2],
            c="red", marker="s", s=100, alpha=0.7, label="GRPO (Ours)"
        )
    
    ax2.set_xlabel("Diversity (Unique Sequences)", fontsize=12)
    ax2.set_ylabel("Q90 Fitness", fontsize=12)
    ax2.set_title("Pareto Frontier: Top Fitness vs Diversity", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[Saved Pareto comparison to {output_path}]")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare SGPO DPO vs Our GRPO Pareto Frontiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--sgpo_results", type=str, default=None,
                        help="Path to SGPO's pareto_summary.csv")
    parser.add_argument("--grpo_results", type=str, default=None,
                        help="Path to our GRPO pareto_summary.csv")
    parser.add_argument("--sgpo_repo", type=str, default=None,
                        help="Path to SGPO repo (auto-finds DPO results)")
    parser.add_argument("--dataset", type=str, default="TrpB",
                        choices=["TrpB", "CreiLOV", "GB1"],
                        help="Dataset name")
    parser.add_argument("--output", type=str, default="outputs/sgpo/pareto_comparison.png",
                        help="Output plot path")
    
    args = parser.parse_args()
    
    # Auto-detect paths
    sgpo_path = args.sgpo_results
    if not sgpo_path and args.sgpo_repo:
        candidates = [
            os.path.join(args.sgpo_repo, "exps", "protein", args.dataset, "causalLM_finetune", "pareto", "DPO", "pareto_summary.csv"),
            os.path.join(args.sgpo_repo, "exps", args.dataset, "causalLM_finetune", "pareto", "DPO", "pareto_summary.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                sgpo_path = c
                break
    
    grpo_path = args.grpo_results
    
    # Load data
    dpo_df = pd.DataFrame()
    grpo_df = pd.DataFrame()
    
    if sgpo_path and os.path.exists(sgpo_path):
        print(f"[Loading SGPO DPO results from {sgpo_path}]")
        dpo_df = load_sgpo_pareto(sgpo_path)
        print(f"  Found {len(dpo_df)} DPO configurations")
    else:
        print("[WARNING] No SGPO DPO results found")
    
    if grpo_path and os.path.exists(grpo_path):
        print(f"[Loading GRPO results from {grpo_path}]")
        grpo_df = load_grpo_pareto(grpo_path)
        print(f"  Found {len(grpo_df)} GRPO configurations")
    else:
        print("[WARNING] No GRPO results found")
    
    if len(dpo_df) == 0 and len(grpo_df) == 0:
        print("[ERROR] No results to compare. Run experiments first:")
        print("  SGPO DPO: cd ~/Code/SGPO && python pareto.py ...")
        print("  Our GRPO: python scripts/run_sgpo_fitness.py --mode pareto ...")
        sys.exit(1)
    
    # Print summary tables
    if len(dpo_df) > 0:
        print("\n[SGPO DPO Results]")
        print(dpo_df.to_string(index=False))
    
    if len(grpo_df) > 0:
        print("\n[Our GRPO Results]")
        # Aggregate across repeats
        agg_cols = ["fitness_scale", "entropy_coef"]
        if all(c in grpo_df.columns for c in agg_cols):
            grpo_agg = grpo_df.groupby(agg_cols).agg({
                "mean_fitness": ["mean", "std"],
                "unique_count": "mean",
                "shannon_entropy": "mean",
            }).reset_index()
            grpo_agg.columns = [f"{a}_{b}" if b else a for a, b in grpo_agg.columns]
            print(grpo_agg.to_string(index=False))
        else:
            print(grpo_df.to_string(index=False))
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Plot comparison
    plot_pareto(dpo_df, grpo_df, args.output)
    
    print("\nDone.")


if __name__ == "__main__":
    main()

