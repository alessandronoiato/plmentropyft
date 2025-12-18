"""
MAFFT Alignment Wrapper

This module provides a wrapper for MAFFT multiple sequence alignment,
used to align generated sequences to the parent (wild-type) sequence.
"""

import os
import subprocess
import tempfile
from typing import List, Optional


class MAFFTAligner:
    """
    Wrapper for MAFFT multiple sequence alignment.
    
    MAFFT is used to align generated sequences to the parent (wild-type) sequence,
    handling insertions/deletions from variable-length generation.
    
    Command: mafft --quiet --add generated.fasta --keeplength parent.fasta > aligned.fasta
    """
    
    def __init__(
        self,
        parent_sequence: str,
        parent_fasta_path: Optional[str] = None,
        mafft_path: str = "mafft",
        tmp_dir: Optional[str] = None,
    ):
        """
        Initialize MAFFT aligner.
        
        Args:
            parent_sequence: Wild-type sequence string
            parent_fasta_path: Path to parent.fasta (will create temp if None)
            mafft_path: Path to MAFFT binary
            tmp_dir: Directory for temporary files
        """
        self.parent_sequence = parent_sequence
        self.mafft_path = mafft_path
        self.tmp_dir = tmp_dir or tempfile.gettempdir()
        
        # Verify MAFFT is available
        self._mafft_available = self._check_mafft()
        
        # Create parent FASTA if not provided
        if parent_fasta_path and os.path.exists(parent_fasta_path):
            self.parent_fasta = parent_fasta_path
            self._owns_parent_fasta = False
        else:
            self.parent_fasta = self._create_parent_fasta()
            self._owns_parent_fasta = True
    
    def _check_mafft(self) -> bool:
        """Check if MAFFT is available."""
        try:
            result = subprocess.run(
                [self.mafft_path, "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"[MAFFTAligner] Warning: MAFFT not found at '{self.mafft_path}'")
            print("  Install with: conda install -c bioconda mafft")
            return False
    
    def _create_parent_fasta(self) -> str:
        """Create temporary FASTA file for parent sequence."""
        path = os.path.join(self.tmp_dir, "sgpo_parent.fasta")
        with open(path, "w") as f:
            f.write(">parent\n")
            f.write(f"{self.parent_sequence}\n")
        return path
    
    @property
    def is_available(self) -> bool:
        """Check if MAFFT is available."""
        return self._mafft_available
    
    def align(self, sequences: List[str]) -> List[str]:
        """
        Align sequences to parent using MAFFT.
        
        Args:
            sequences: List of generated sequences (variable length OK)
        
        Returns:
            List of aligned sequences (same length as parent)
        """
        if not sequences:
            return []
        
        if not self._mafft_available:
            # Fallback: pad/truncate to parent length
            print("[MAFFTAligner] MAFFT unavailable, using simple truncation")
            return self._fallback_align(sequences)
        
        # Create temp FASTA for generated sequences
        gen_fasta = os.path.join(self.tmp_dir, "sgpo_generated.fasta")
        aligned_fasta = os.path.join(self.tmp_dir, "sgpo_aligned.fasta")
        
        with open(gen_fasta, "w") as f:
            for i, seq in enumerate(sequences):
                f.write(f">{i}\n{seq}\n")
        
        try:
            # Run MAFFT
            # mafft --quiet --add generated.fasta --keeplength parent.fasta > aligned.fasta
            cmd = [
                self.mafft_path,
                "--quiet",
                "--add", gen_fasta,
                "--keeplength", self.parent_fasta
            ]
            
            with open(aligned_fasta, "w") as out_file:
                result = subprocess.run(
                    cmd,
                    stdout=out_file,
                    stderr=subprocess.PIPE,
                    timeout=300  # 5 min timeout
                )
            
            if result.returncode != 0:
                print(f"[MAFFTAligner] MAFFT error: {result.stderr.decode()}")
                return self._fallback_align(sequences)
            
            # Parse aligned sequences
            aligned = self._parse_fasta(aligned_fasta)
            
            # Remove parent sequence (first one) if present
            if aligned and len(aligned) > len(sequences):
                aligned = aligned[1:]  # Skip parent
            
            return aligned
            
        except subprocess.TimeoutExpired:
            print("[MAFFTAligner] MAFFT timeout, using fallback")
            return self._fallback_align(sequences)
        except Exception as e:
            print(f"[MAFFTAligner] Error: {e}")
            return self._fallback_align(sequences)
        finally:
            # Cleanup temp files
            for f in [gen_fasta, aligned_fasta]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
    
    def _parse_fasta(self, path: str) -> List[str]:
        """Parse FASTA file and return sequences."""
        sequences = []
        current_seq = []
        
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_seq:
                            sequences.append("".join(current_seq))
                            current_seq = []
                    else:
                        current_seq.append(line)
                if current_seq:
                    sequences.append("".join(current_seq))
        except Exception as e:
            print(f"[MAFFTAligner] Error parsing FASTA: {e}")
        
        return sequences
    
    def _fallback_align(self, sequences: List[str]) -> List[str]:
        """Simple fallback when MAFFT is unavailable."""
        parent_len = len(self.parent_sequence)
        aligned = []
        for seq in sequences:
            if len(seq) >= parent_len:
                aligned.append(seq[:parent_len])
            else:
                # Pad with parent sequence
                aligned.append(seq + self.parent_sequence[len(seq):])
        return aligned
    
    def __del__(self):
        """Cleanup temporary parent FASTA."""
        if hasattr(self, '_owns_parent_fasta') and self._owns_parent_fasta:
            if hasattr(self, 'parent_fasta') and os.path.exists(self.parent_fasta):
                try:
                    os.remove(self.parent_fasta)
                except OSError:
                    pass

