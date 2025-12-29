"""
SGPO Fitness Oracle

This module provides wrappers for the SGPO fitness oracle ensemble,
which predicts protein fitness from sequence.
"""

import os
import random
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .data import AA_ALPHABET


def _is_verbose() -> bool:
    """Check if verbose/debug output is enabled."""
    return os.environ.get("SGPO_VERBOSE", "0") == "1"

if TYPE_CHECKING:
    from .projector import TrpBProjector


class SGPOOracleModel(nn.Module):
    """
    SGPO Oracle MLP architecture.
    
    From SGPO train_oracle.py:
    - Input: flattened one-hot encoding (seq_len * 20)
    - Hidden: 400 units, ReLU, Dropout(0.1)
    - Output: 1 (fitness score)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 400,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SGPOFitnessOracle:
    """
    Wrapper for SGPO fitness oracle ensemble.
    
    The SGPO oracle is an ensemble of MLPs trained on experimental fitness data.
    
    IMPORTANT: The oracle expects **fixed-length sequences** corresponding to
    mutated positions only (e.g., 15 positions for TrpB), NOT full protein sequences.
    
    For TrpB:
        - Input: 15 amino acids (the "Combo" column)
        - Fitness range: [0.0, 2.26], mean=0.056
    
    The oracle can optionally apply a Hamming distance penalty to discourage
    sequences that are too different from the parent.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        dataset: str = "TrpB",
        hidden_dim: int = 400,
        dropout_rate: float = 0.1,
        parent_combo: Optional[str] = None,
        impose_penalty: bool = False,
        penalty_cutoff: int = 233,
        penalty_rate: float = 0.99,
    ):
        """
        Initialize oracle.
        
        Args:
            checkpoint_dir: Path to oracle checkpoints (e.g., oracle/checkpoints/TrpB/)
            device: Device for inference
            dataset: Dataset name (TrpB, CreiLOV, GB1)
            hidden_dim: Hidden layer dimension (default: 400)
            dropout_rate: Dropout rate (default: 0.1)
            parent_combo: Wild-type combo sequence (for penalty)
            impose_penalty: Whether to apply Hamming distance penalty
            penalty_cutoff: Hamming distance threshold before penalty kicks in
            penalty_rate: Exponential decay rate for penalty
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.models: List[SGPOOracleModel] = []
        self.input_dim = None
        
        # Penalty settings
        self.parent_combo = parent_combo
        self.impose_penalty = impose_penalty
        self.penalty_cutoff = penalty_cutoff
        self.penalty_rate = penalty_rate
        
        # Dataset-specific sequence lengths
        self.seq_lengths = {
            "TrpB": 15,
            "CreiLOV": 112,
            "GB1": 56,
        }
        
        # Load ensemble if checkpoint dir exists
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            self._load_ensemble(checkpoint_dir)
        else:
            print(f"[SGPOFitnessOracle] No checkpoint dir at {checkpoint_dir}, using placeholder.")
    
    def _load_ensemble(self, checkpoint_dir: str):
        """Load all model checkpoints in the directory."""
        # Get all .pth files
        ckpt_files = sorted([
            f for f in os.listdir(checkpoint_dir) 
            if f.endswith('.pth') or f.endswith('.pt')
        ])
        
        if not ckpt_files:
            print(f"[SGPOFitnessOracle] No checkpoint files found in {checkpoint_dir}")
            return
        
        # Infer input dimension from sequence length
        seq_len = self.seq_lengths.get(self.dataset, 15)
        self.input_dim = seq_len * 20  # One-hot encoding
        
        print(f"[SGPOFitnessOracle] Loading {len(ckpt_files)} models from {checkpoint_dir}")
        print(f"  Dataset: {self.dataset}, Seq length: {seq_len}, Input dim: {self.input_dim}")
        
        for ckpt_file in ckpt_files:
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            
            # DEBUG: Check actual model dimensions from checkpoint
            if not hasattr(self, '_dim_debug_shown'):
                self._dim_debug_shown = True
                fc1_weight = state_dict.get('fc1.weight', None)
                if fc1_weight is not None:
                    actual_input_dim = fc1_weight.shape[1]
                    if _is_verbose():
                        print(f"[DEBUG Oracle] Checkpoint fc1.weight shape: {fc1_weight.shape}")
                        print(f"[DEBUG Oracle] Expected input_dim={self.input_dim}, actual from checkpoint={actual_input_dim}")
                    if actual_input_dim != self.input_dim:
                        print(f"[WARNING] Input dimension mismatch! Using checkpoint's input_dim={actual_input_dim}")
                        self.input_dim = actual_input_dim
            
            model = SGPOOracleModel(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
        
        print(f"[SGPOFitnessOracle] Loaded {len(self.models)} ensemble models")
    
    def _encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode sequences as flattened one-hot vectors.
        
        Args:
            sequences: List of amino acid sequences (must be same length!)
            
        Returns:
            Tensor of shape (batch, seq_len * 20)
        """
        aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
        batch_size = len(sequences)
        seq_len = len(sequences[0]) if sequences else 0
        
        # DEBUG: Check expected vs actual length
        expected_len = self.seq_lengths.get(self.dataset, 15)
        if not hasattr(self, '_len_debug_shown') and _is_verbose():
            self._len_debug_shown = True
            print(f"[DEBUG Oracle] Expected seq_len={expected_len}, got seq_len={seq_len}")
            print(f"[DEBUG Oracle] Sample sequences: {sequences[:3]}")
        
        # Validate all sequences have same length
        for s in sequences:
            if len(s) != seq_len:
                raise ValueError(f"All sequences must have same length. Expected {seq_len}, got {len(s)}")
        
        # One-hot encoding: (batch, seq_len, 20) then flatten to (batch, seq_len * 20)
        encoded = torch.zeros(batch_size, seq_len, 20, device=self.device)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in aa_to_idx:
                    encoded[i, j, aa_to_idx[aa]] = 1.0
                # Unknown AAs get zero vector (handled gracefully)
        
        return encoded.view(batch_size, -1)  # Flatten
    
    def _hamming_distance(self, s1: str, s2: str) -> int:
        """Compute Hamming distance between two sequences."""
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    def _apply_penalty(self, predictions: np.ndarray, sequences: List[str]) -> np.ndarray:
        """
        Apply Hamming distance penalty to predictions.
        
        From SGPO: penalty = rate^(max(0, hamming - cutoff))
        """
        if not self.impose_penalty or self.parent_combo is None:
            return predictions
        
        penalties = []
        for seq in sequences:
            hd = self._hamming_distance(self.parent_combo, seq)
            if hd <= self.penalty_cutoff:
                penalties.append(1.0)
            else:
                penalties.append(self.penalty_rate ** (hd - self.penalty_cutoff))
        
        return predictions * np.array(penalties)
    
    def __call__(self, sequences: List[str]) -> List[float]:
        """
        Compute fitness scores for a batch of sequences.
        
        Args:
            sequences: List of protein sequences (amino acid strings)
                       Must be the correct length for the dataset (e.g., 15 for TrpB)
            
        Returns:
            List of fitness scores (higher = better, typically in [0, ~2.3] range)
        """
        if not self.models:
            # Placeholder: return random fitness for testing
            print("[SGPOFitnessOracle] Warning: No models loaded, returning random scores")
            return [random.random() for _ in sequences]
        
        if not sequences:
            return []
        
        # Encode sequences
        encoded = self._encode_sequences(sequences)
        
        # Run through ensemble and average predictions
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                preds = model(encoded).squeeze(-1)  # (batch,)
                all_preds.append(preds)
        
        # Average across ensemble
        ensemble_preds = torch.stack(all_preds, dim=0).mean(dim=0)  # (batch,)
        predictions = ensemble_preds.cpu().numpy()
        
        # DEBUG: Show raw predictions (first call only, when verbose)
        if not hasattr(self, '_debug_shown') and _is_verbose():
            self._debug_shown = True
            print(f"[DEBUG Oracle] Raw predictions (first 5): {predictions[:5]}")
            print(f"[DEBUG Oracle] Min={predictions.min():.4f}, Max={predictions.max():.4f}, Mean={predictions.mean():.4f}")
        
        # Apply Hamming penalty if enabled
        predictions = self._apply_penalty(predictions, sequences)
        
        # Note: We do NOT clamp to non-negative here.
        # Raw predictions (including negatives) preserve gradient signal for GRPO.
        # SGPO's original code clamped because "there shouldn't be any for CreiLOV",
        # but for GRPO training we need the relative ordering of all predictions.
        
        return predictions.tolist()
    
    def score_combo_sequences(self, combo_sequences: List[str]) -> List[float]:
        """
        Score sequences that are already in "Combo" format (mutated positions only).
        
        This is the native format expected by the oracle.
        """
        return self(combo_sequences)
    
    def score_full_sequences(
        self,
        full_sequences: List[str],
        projector: "TrpBProjector",
    ) -> Tuple[List[float], List[str]]:
        """
        Score full protein sequences by projecting to combo first.
        
        Args:
            full_sequences: List of full-length sequences (389aa for TrpB)
            projector: TrpBProjector instance
        
        Returns:
            (scores, combos): Fitness scores and extracted combo sequences
        """
        _, combos = projector.project(full_sequences)
        scores = self(combos)
        return scores, combos

