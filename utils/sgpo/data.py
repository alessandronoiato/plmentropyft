"""
TrpB Dataset Constants and Data Loading Utilities

This module contains constants for the TrpB dataset and utilities for
loading fitness data from SGPO-formatted CSV files.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# TrpB Dataset Constants
# =============================================================================

TRPB_PARENT_SEQUENCE = (
    "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGK"
    "TRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIV"
    "RNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSV"
    "SAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIR"
)

# 15 positions (1-indexed) that form the "Combo" sequence in SGPO
TRPB_COMBO_POSITIONS = [117, 118, 119, 162, 166, 182, 183, 184, 185, 186, 227, 228, 230, 231, 301]

# Wild-type combo sequence (15 amino acids at the mutated positions)
TRPB_WT_COMBO = "TAALIYVFGSVSGSY"

# Standard 20 amino acids
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

TRPB_INFO = """
TrpB (Tryptophan Synthase Beta Subunit) Dataset
================================================

Source: Wu et al. (2019) - Combinatorial mutagenesis study

Key characteristics:
- Wild-type: Pyrococcus furiosus TrpB (389 aa)
- Mutated positions: 15 non-contiguous positions
- Positions (1-indexed): 117, 118, 119, 162, 166, 182, 183, 184, 185, 186, 227, 228, 230, 231, 301
- Wild-type "Combo": TAALIYVFGSVSGSY
- Fitness metric: Catalytic activity relative to WT (WT = 1.0)
- Dataset size: 111,883 variants

Fitness statistics:
- Range: [0.0, 2.26] (>1.0 means improved over WT!)
- Mean: 0.056
- Std: 0.147
- Median: 0.018
- P90: 0.11, P95: 0.20, P99: 0.87

Data format (fitness.csv):
- Combo: 15-char sequence of mutated positions
- mut: Mutation notation (e.g., T117A:A119C)
- n_mut: Number of mutations (0-4)
- fitness: Relative fitness (WT = 1.0)
- split: train/test/validation

For more details, see:
- SGPO repo: https://github.com/jsunn-y/SGPO/tree/main/data
- Original paper: Wu et al., Nature Communications (2019)
"""


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_trpb_fitness_data(
    data_path: str,
    split: Optional[str] = None,
) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
    """
    Load TrpB fitness data from SGPO-formatted CSV.
    
    SGPO format:
        Combo,mut,n_mut,fitness,split
        AAALIYVFGSVSGSY,T117A,1,0.1479...,train
        ...
    
    Args:
        data_path: Path to fitness.csv
        split: Optional filter for split (train/test/validation)
    
    Returns:
        (combo_sequences, fitness_values, full_records)
    """
    sequences = []
    fitness_values = []
    records = []
    
    if not os.path.exists(data_path):
        print(f"[load_trpb_fitness_data] File not found: {data_path}")
        return sequences, fitness_values, records
    
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combo = row.get("Combo", "")
            fit_str = row.get("fitness", "")
            row_split = row.get("split", "")
            
            # Filter by split if specified
            if split and row_split != split:
                continue
            
            if combo and fit_str:
                try:
                    fit = float(fit_str)
                    sequences.append(combo)
                    fitness_values.append(fit)
                    records.append({
                        "combo": combo,
                        "mut": row.get("mut", ""),
                        "n_mut": int(row.get("n_mut", 0)),
                        "fitness": fit,
                        "split": row_split,
                    })
                except ValueError:
                    continue
    
    print(f"[load_trpb_fitness_data] Loaded {len(sequences)} sequences" + 
          (f" (split={split})" if split else ""))
    if fitness_values:
        print(f"  Fitness range: [{min(fitness_values):.4f}, {max(fitness_values):.4f}]")
        print(f"  Fitness mean: {np.mean(fitness_values):.4f}, std: {np.std(fitness_values):.4f}")
    
    return sequences, fitness_values, records


# =============================================================================
# Sequence Conversion Utilities
# =============================================================================

def combo_to_full_sequence(
    combo: str,
    positions: Optional[List[int]] = None,
    parent: Optional[str] = None,
) -> str:
    """
    Convert a 15-char combo sequence back to full protein sequence.
    
    Args:
        combo: 15-character combo sequence
        positions: List of 1-indexed positions (default: TRPB_COMBO_POSITIONS)
        parent: Parent sequence to use as template (default: TRPB_PARENT_SEQUENCE)
    
    Returns:
        Full protein sequence with mutations applied
    """
    if positions is None:
        positions = TRPB_COMBO_POSITIONS
    if parent is None:
        parent = TRPB_PARENT_SEQUENCE
    
    if len(combo) != len(positions):
        raise ValueError(f"Combo length {len(combo)} != positions length {len(positions)}")
    
    # Convert parent to list for mutation
    full_seq = list(parent)
    
    # Apply mutations from combo
    for i, pos in enumerate(positions):
        full_seq[pos - 1] = combo[i]  # Convert to 0-indexed
    
    return "".join(full_seq)


def full_sequence_to_combo(
    full_seq: str,
    positions: Optional[List[int]] = None,
) -> str:
    """
    Extract combo sequence from full protein sequence.
    
    Args:
        full_seq: Full protein sequence
        positions: List of 1-indexed positions (default: TRPB_COMBO_POSITIONS)
    
    Returns:
        15-character combo sequence
    """
    if positions is None:
        positions = TRPB_COMBO_POSITIONS
    
    return "".join(full_seq[pos - 1] for pos in positions)


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_fitness_distribution(
    fitness_values: List[float],
    name: str = "fitness",
) -> Dict[str, Any]:
    """
    Analyze and print fitness distribution statistics.
    
    Args:
        fitness_values: List of fitness scores
        name: Label for the distribution
    
    Returns:
        Dictionary of statistics
    """
    if not fitness_values:
        print(f"[{name}] No data to analyze")
        return {}
    
    arr = np.array(fitness_values)
    stats = {
        "count": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "q90": float(np.percentile(arr, 90)),
        "q99": float(np.percentile(arr, 99)),
    }
    
    print(f"\n[{name}] Distribution Statistics:")
    print(f"  Count:  {stats['count']}")
    print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean:   {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Q25/Q75: {stats['q25']:.4f} / {stats['q75']:.4f}")
    print(f"  Q90/Q99: {stats['q90']:.4f} / {stats['q99']:.4f}")
    
    return stats

