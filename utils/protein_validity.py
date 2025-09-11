from typing import Dict, Tuple
import math
import re


# Background amino-acid frequencies (approximate BLOSUM-like background)
_AA_FREQ: Dict[str, float] = {
    "A": 0.0806,
    "R": 0.0564,
    "N": 0.0406,
    "D": 0.0542,
    "C": 0.0133,
    "Q": 0.0393,
    "E": 0.0672,
    "G": 0.0707,
    "H": 0.0227,
    "I": 0.0593,
    "L": 0.0965,
    "K": 0.0581,
    "M": 0.0241,
    "F": 0.0386,
    "P": 0.0470,
    "S": 0.0660,
    "T": 0.0535,
    "W": 0.0108,
    "Y": 0.0292,
    "V": 0.0687,
}

_AA_SET = set(_AA_FREQ.keys())


# --- New 0/1 validity oracle components ---
# Canonical AA20 are in _AA_SET. Optionally allow 'U' and 'Ċ'.
_POLAR_SET = set("STNQYCH")
_HYDRO_SET = set("AILMVFWPG")


def is_valid_basic(seq: str, min_len: int = 30, max_len: int = 2048, allow_U: bool = False, allow_Cdot: bool = False, max_run: int = 6) -> int:
    """Binary validity oracle.

    Rules:
      - Alphabet-only: 20 canonical AAs; optionally allow 'U' and 'Ċ'
      - No pathological runs: reject if any residue repeats > max_run times
      - Composition sanity: require at least one polar and one hydrophobic residue
      - Length within [min_len, max_len]
    Returns 1 if valid else 0.
    """
    if seq is None:
        return 0
    s = str(seq).strip().upper()
    if not (min_len <= len(s) <= max_len):
        return 0
    alphabet = set(_AA_SET)
    if allow_U:
        alphabet.add("U")
    if allow_Cdot:
        alphabet.add("Ċ")
    # Alphabet constraint
    for ch in s:
        if ch not in alphabet:
            return 0
    # Pathological homopolymer runs
    if re.search(rf"(.)\1{{{max_run},}}", s) is not None:
        return 0
    # Composition sanity
    has_polar = len(set(s) & _POLAR_SET) > 0
    has_hydro = len(set(s) & _HYDRO_SET) > 0
    if not (has_polar and has_hydro):
        return 0
    return 1


def compute_validity_score(sequence: str) -> float:
    """Deprecated: kept for compatibility. Always returns 0.0. Use is_valid_basic."""
    return 0.0


def compute_composition_l1(sequence: str) -> Tuple[float, float]:
    """Deprecated: kept for compatibility. Returns (0.0, 2.0)."""
    return 0.0, 2.0


def max_homopolymer_run(sequence: str) -> int:
    """Deprecated: kept for compatibility. Use is_valid_basic instead."""
    return 0


# --- Empirical property helpers ---

# Kyte-Doolittle hydropathy index (AA -> value)
_HYDROPATHY: Dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

_HYDROPHOBIC_SET = {aa for aa, v in _HYDROPATHY.items() if v > 0.0}
_CHARGED_SET = set("KRHDE")  # basic+acidic as a coarse proxy


def longest_run_in_set(sequence: str, allowed: set) -> int:
    """Deprecated: kept for compatibility. Use is_valid_basic instead."""
    return 0


def mean_hydropathy(sequence: str) -> float:
    """Deprecated: kept for compatibility. Use is_valid_basic instead."""
    return 0.0


# pKa values (approx) for termini and side chains
_PKA = {
    "N_TERMINUS": 9.69,
    "C_TERMINUS": 2.34,
    "K": 10.54,
    "R": 12.48,
    "H": 6.04,
    "D": 3.90,
    "E": 4.07,
    "C": 8.18,
    "Y": 10.46,
}


def _net_charge_at_pH(sequence: str, pH: float) -> float:
    """Deprecated: kept for compatibility. Use is_valid_basic instead."""
    return 0.0


def compute_isoelectric_point(sequence: str) -> float:
    """Deprecated: kept for compatibility. Use is_valid_basic instead."""
    return 7.0


_MOTIF_PATTERNS = [
    re.compile(r"N[^P][ST][^P]"),  # N-glycosylation motif
    re.compile(r"[ST].{2}[DE]"),   # CK2-like
    re.compile(r"[ST].{1}[RK]"),   # PKC-like
    re.compile(r"P..P"),           # Proline-rich motif
    re.compile(r"C..C"),           # Cys pair (potential disulfide spacing)
]


def count_motif_hits(sequence: str) -> int:
    """Deprecated: kept for compatibility. Use is_valid_basic instead."""
    return 0


def compute_validity_score_enhanced(sequence: str) -> Tuple[float, Dict[str, float]]:
    """Deprecated: kept for compatibility. Returns (0.0, {})."""
    return 0.0, {}

