from typing import List, Dict, Tuple

AMINO_ACIDS: Tuple[str, ...] = (
	"A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
	"M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)


def get_amino_acid_token_ids(tokenizer) -> List[int]:
    """Extract token ids for the 20 standard amino acids as single tokens.

    Strategy (per AA):
    1) Prefer direct encoding without specials; must yield exactly one id.
    2) Fallback: tokenizer.tokenize(letter) must yield one token; convert to id.
    3) Fallback: convert_tokens_to_ids(letter) if valid id.

    Raises ValueError if any AA cannot be mapped to a single token id or if
    uniqueness across all 20 residues is not satisfied.
    """
    ids: List[int] = []
    missing: List[str] = []
    for aa in AMINO_ACIDS:
        # Prefer direct encode path
        try:
            enc = tokenizer(aa, add_special_tokens=False).get("input_ids", [])
        except Exception:
            enc = []
        if isinstance(enc, list) and len(enc) == 1 and isinstance(enc[0], int):
            ids.append(enc[0])
            continue
        # Fallback: single token from tokenize
        try:
            toks = tokenizer.tokenize(aa)
        except Exception:
            toks = []
        if isinstance(toks, list) and len(toks) == 1:
            tid = tokenizer.convert_tokens_to_ids(toks[0])
            if isinstance(tid, int) and tid >= 0:
                ids.append(tid)
                continue
        # Fallback: direct vocab id
        tid = tokenizer.convert_tokens_to_ids(aa)
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
            continue
        missing.append(aa)

    if missing:
        raise ValueError(f"Tokenizer does not expose single-token ids for amino acids: {missing}")

    # Ensure exactly 20 unique ids (no collisions across residues)
    seen: Dict[int, None] = {}
    uniq: List[int] = []
    for tid in ids:
        if tid not in seen:
            seen[tid] = None
            uniq.append(tid)
    if len(uniq) != len(AMINO_ACIDS):
        raise ValueError(
            f"Expected 20 unique amino-acid token ids, got {len(uniq)}. "
            f"This tokenizer may not represent each residue as a single token."
        )
    return uniq
