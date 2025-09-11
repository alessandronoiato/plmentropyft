from typing import Dict, List

import torch
from transformers import LogitsProcessor


class ProteinPieceLogitsProcessor(LogitsProcessor):
    def __init__(self, id_to_len: Dict[int, int], eos_id: int, horizon: int, tokenizer):
        super().__init__()
        self.id_to_len = id_to_len
        self.eos_id = eos_id
        self.horizon = horizon
        self.tok = tokenizer
        self._allow_by_remaining = None  # lazy (horizon+1, vocab)

    def _ids_to_pieces(self, ids: List[int]) -> List[int]:
        # Filter out BOS/EOS/PAD and any non-AA-piece token (must be in id_to_len)
        specials = {self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id}
        return [i for i in ids if i not in specials and i in self.id_to_len]

    def _ensure_precomputed(self, vocab: int, device: torch.device):
        if self._allow_by_remaining is not None and self._allow_by_remaining.size(1) == vocab:
            return
        # Build boolean allow masks for each remaining length 0..horizon
        allow = torch.zeros((self.horizon + 1, vocab), dtype=torch.bool, device=device)
        # remaining == 0: only EOS
        if 0 <= self.eos_id < vocab:
            allow[0, self.eos_id] = True
        # remaining >= 1: any AA piece whose length <= remaining
        for rem in range(1, self.horizon + 1):
            for tid, length in self.id_to_len.items():
                if 0 <= tid < vocab and length <= rem:
                    allow[rem, tid] = True
        # Never allow PAD
        if self.tok.pad_token_id is not None and 0 <= self.tok.pad_token_id < vocab:
            allow[:, self.tok.pad_token_id] = False
        self._allow_by_remaining = allow

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch, vocab = scores.shape
        self._ensure_precomputed(vocab, scores.device)
        for b in range(batch):
            ids = input_ids[b].tolist()
            prefix = self._ids_to_pieces(ids)
            # Residue budget
            consumed = sum(self.id_to_len.get(t, 0) for t in prefix)
            remaining = max(0, self.horizon - consumed)
            rem = remaining if remaining <= self.horizon else self.horizon
            allow_row = self._allow_by_remaining[rem]
            mask = ~allow_row
            scores[b][mask] = -1e9
        return scores


