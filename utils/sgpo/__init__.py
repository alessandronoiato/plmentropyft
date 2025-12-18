"""
SGPO (Steering Generative Models for Protein Optimization) Integration

This package provides components for integrating with the SGPO framework
for protein fitness optimization.

Components:
- ProGen2Wrapper: Load and use ProGen2 models for sequence generation
- MAFFTAligner: Align generated sequences to parent using MAFFT
- TrpBProjector: Project sequences to combo format for oracle scoring
- SGPOFitnessOracle: Score sequences with the SGPO fitness oracle ensemble
- SGPOPipeline: Complete pipeline orchestration
- make_fitness_reward: Create reward function for GRPO training

Usage:
    from utils.sgpo import (
        ProGen2Wrapper,
        MAFFTAligner,
        TrpBProjector,
        SGPOFitnessOracle,
        SGPOPipeline,
        make_fitness_reward,
        TRPB_PARENT_SEQUENCE,
        TRPB_COMBO_POSITIONS,
        TRPB_WT_COMBO,
    )
"""

# Data constants and utilities
from .data import (
    AA_ALPHABET,
    TRPB_COMBO_POSITIONS,
    TRPB_INFO,
    TRPB_PARENT_SEQUENCE,
    TRPB_WT_COMBO,
    analyze_fitness_distribution,
    combo_to_full_sequence,
    full_sequence_to_combo,
    load_trpb_fitness_data,
)

# Model wrapper
from .model import ProGen2Wrapper

# Alignment
from .alignment import MAFFTAligner

# Projection
from .projector import TrpBProjector

# Oracle
from .oracle import SGPOFitnessOracle, SGPOOracleModel

# Pipeline and reward
from .pipeline import SGPOPipeline, make_fitness_reward

__all__ = [
    # Constants
    "AA_ALPHABET",
    "TRPB_COMBO_POSITIONS",
    "TRPB_INFO",
    "TRPB_PARENT_SEQUENCE",
    "TRPB_WT_COMBO",
    # Data utilities
    "analyze_fitness_distribution",
    "combo_to_full_sequence",
    "full_sequence_to_combo",
    "load_trpb_fitness_data",
    # Model
    "ProGen2Wrapper",
    # Alignment
    "MAFFTAligner",
    # Projection
    "TrpBProjector",
    # Oracle
    "SGPOFitnessOracle",
    "SGPOOracleModel",
    # Pipeline
    "SGPOPipeline",
    "make_fitness_reward",
]

