from dataclasses import dataclass
from typing import List


@dataclass
class ProteinConfig:
	"""Configuration for protein environment.

	- horizon: maximum number of amino-acid tokens before EOS is allowed.
	"""
	horizon: int = 128


class ProteinEnv:
	"""Legality for protein sequences:

	- Before horizon: only amino-acid tokens are allowed
	- At or after horizon: only EOS is allowed
	- PAD/BOS are never allowed to be generated
	"""

	def __init__(self, config: ProteinConfig):
		self.config = config

	def legal_action_ids(self, prefix: List[int], aa_ids: List[int], eos_id: int) -> List[int]:
		"""Return allowed token ids for the next action given the current prefix.

		prefix contains previously generated action token ids (AA ids), i.e., special tokens are not included.
		"""
		t = len(prefix)
		if t >= self.config.horizon:
			return [eos_id]
		return list(aa_ids)
