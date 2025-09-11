from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProteinPieceConfig:
    """Residue-length aware environment for AA-only BPE pieces.

    horizon: total number of residues required before EOS.
    id_to_len: mapping token_id -> number of residues contributed (only for AA-only pieces).
    """
    horizon: int = 128
    id_to_len: Dict[int, int] = None  # to be provided


class ProteinPieceEnv:
    def __init__(self, config: ProteinPieceConfig):
        if config.id_to_len is None:
            raise ValueError("ProteinPieceEnv requires id_to_len for AA-only pieces")
        self.config = config

    def legal_action_ids(self, prefix: List[int], aa_ids_unused, eos_id: int) -> List[int]:
        """Return allowed token ids for the next action given the current prefix.

        prefix is the list of previously generated AA-piece token ids (no specials).
        Allow any AA piece whose aa_length <= remaining residues; allow EOS only when remaining == 0.
        """
        consumed = 0
        for tid in prefix:
            consumed += self.config.id_to_len.get(tid, 0)
        remaining = max(0, self.config.horizon - consumed)
        if remaining == 0:
            return [eos_id]
        allow: List[int] = []
        for tid, length in self.config.id_to_len.items():
            if length <= remaining:
                allow.append(tid)
        return allow


