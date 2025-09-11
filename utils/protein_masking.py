from typing import List
import torch
from transformers import LogitsProcessor


class ProteinLogitsProcessor(LogitsProcessor):
	def __init__(self, aa_ids: List[int], eos_id: int, horizon: int, tokenizer):
		super().__init__()
		self.aa_ids = set(aa_ids)
		self.eos_id = eos_id
		self.horizon = horizon
		self.tok = tokenizer

	def _ids_to_actions(self, ids: List[int]) -> List[int]:
		return [i for i in ids if i not in {self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id}]

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
		batch, vocab = scores.shape
		for b in range(batch):
			ids = input_ids[b].tolist()
			prefix = self._ids_to_actions(ids)
			allow = torch.zeros(vocab, dtype=torch.bool, device=scores.device)
			if len(prefix) >= self.horizon:
				allow[self.eos_id] = True
			else:
				for a in self.aa_ids:
					allow[a] = True
			mask = ~allow
			# Never force-mask PAD if it equals EOS; otherwise we might mask the only allowed token at horizon
			if self.tok.pad_token_id is not None and self.tok.pad_token_id != self.eos_id:
				mask[self.tok.pad_token_id] = True
			scores[b][mask] = -1e9
		return scores
