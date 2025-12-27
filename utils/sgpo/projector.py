"""
TrpB Sequence Projector

This module projects aligned sequences to the 15-position combo format
expected by the SGPO fitness oracle.
"""

import random
from typing import List, Tuple

from .data import AA_ALPHABET


class TrpBProjector:
    """
    Project aligned sequences to the 15-position combo format expected by the oracle.
    
    Following SGPO's DPO.py::project_sequences:
    1. Replace gaps ('-') with WT residue
    2. Replace all non-mutated positions with WT
    3. Replace invalid characters with random AA
    4. Extract 15-char combo from the 15 mutated positions
    """
    
    def __init__(
        self,
        parent_sequence: str,
        mutated_positions: List[int],  # 1-indexed
    ):
        """
        Initialize projector.
        
        Args:
            parent_sequence: Wild-type full sequence
            mutated_positions: List of 1-indexed positions that are mutated
        """
        self.parent_sequence = parent_sequence
        self.positions = mutated_positions
        self.positions_0idx = [p - 1 for p in mutated_positions]  # Convert to 0-indexed
    
    def project(self, aligned_sequences: List[str]) -> Tuple[List[str], List[str]]:
        """
        Project aligned sequences to combo format.
        
        Args:
            aligned_sequences: List of aligned sequences (same length as parent)
        
        Returns:
            (full_projected, combos): Tuple of projected full sequences and 15-char combos
        """
        full_projected = []
        combos = []
        
        for seq in aligned_sequences:
            # Step 1: Replace gaps with WT
            seq_list = list(seq)
            for i, char in enumerate(seq_list):
                if char == '-' and i < len(self.parent_sequence):
                    seq_list[i] = self.parent_sequence[i]
            
            # Step 2: Replace non-mutated positions with WT
            for i in range(len(seq_list)):
                if i not in self.positions_0idx and i < len(self.parent_sequence):
                    seq_list[i] = self.parent_sequence[i]
            
            # Step 3: Replace invalid characters
            for i, char in enumerate(seq_list):
                if char not in AA_ALPHABET:
                    seq_list[i] = random.choice(AA_ALPHABET)
            
            projected = "".join(seq_list)
            full_projected.append(projected)
            
            # Step 4: Extract combo
            combo = "".join(projected[p] for p in self.positions_0idx if p < len(projected))
            
            # DEBUG: Show combo extraction on first call
            if not hasattr(self, '_combo_debug_shown'):
                self._combo_debug_shown = True
                print(f"[DEBUG Projector] Projected len={len(projected)}, positions_0idx={self.positions_0idx[:5]}...")
                print(f"[DEBUG Projector] Extracted combo len={len(combo)}, combo='{combo}'")
            
            combos.append(combo)
        
        return full_projected, combos
    
    def combo_to_full(self, combo: str) -> str:
        """
        Convert combo back to full sequence.
        
        Args:
            combo: 15-character combo sequence
        
        Returns:
            Full protein sequence with combo mutations applied
        """
        seq = list(self.parent_sequence)
        for i, pos in enumerate(self.positions_0idx):
            if i < len(combo):
                seq[pos] = combo[i]
        return "".join(seq)
    
    def full_to_combo(self, full_seq: str) -> str:
        """
        Extract combo from full sequence.
        
        Args:
            full_seq: Full protein sequence
        
        Returns:
            15-character combo sequence
        """
        return "".join(full_seq[p] for p in self.positions_0idx if p < len(full_seq))

