from typing import Dict, List, Tuple

import re


AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def is_aa_only(text: str) -> bool:
    return bool(AA_RE.match(text))


def build_aa_piece_table(tokenizer) -> Tuple[Dict[int, int], List[int]]:
    """Return mapping: token_id -> aa_length for AA-only tokens, and a list of those ids.

    - Decodes each vocab id to text and checks that it is strictly letters in the 20 AA set.
    - The aa_length is len(text).
    """
    id_to_len: Dict[int, int] = {}
    ids: List[int] = []
    # Some tokenizers don't expose vocab size directly; derive from added tokens + known length
    try:
        vocab_size = len(tokenizer)
    except Exception:
        vocab_size = tokenizer.vocab_size
    for tid in range(vocab_size):
        try:
            tok = tokenizer.convert_ids_to_tokens([tid])[0]
            # Strip common BPE leading markers if present (e.g., 'Ġ') by decoding back
            text = tokenizer.convert_tokens_to_string([tok])
            text = text.strip()
            if text and is_aa_only(text):
                id_to_len[tid] = len(text)
                ids.append(tid)
        except Exception:
            continue
    return id_to_len, ids


