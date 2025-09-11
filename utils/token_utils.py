from typing import List, Dict, Tuple

AMINO_ACIDS: Tuple[str, ...] = (
	"A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
	"M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)


def get_amino_acid_token_ids(tokenizer) -> List[int]:
	"""Best-effort extraction of token ids for the 20 standard amino acids.

	Tries, in order, for each letter:
	1) tokenizer.convert_tokens_to_ids(letter)
	2) tokenizer.convert_tokens_to_ids(" " + letter)
	3) tokenizer.tokenize(letter) -> single token -> convert to id
	4) tokenizer(letter, add_special_tokens=False)['input_ids'] and take the first id if length==1

	Raises ValueError if an id cannot be determined for any amino acid.
	"""
	ids: List[int] = []
	fail: List[str] = []
	for aa in AMINO_ACIDS:
		id_try = tokenizer.convert_tokens_to_ids(aa)
		if isinstance(id_try, int) and id_try >= 0:
			ids.append(id_try)
			continue
		id_try = tokenizer.convert_tokens_to_ids(" " + aa)
		if isinstance(id_try, int) and id_try >= 0:
			ids.append(id_try)
			continue
		# Try tokenize path
		toks = tokenizer.tokenize(aa)
		if isinstance(toks, list) and len(toks) == 1:
			id_try = tokenizer.convert_tokens_to_ids(toks[0])
			if isinstance(id_try, int) and id_try >= 0:
				ids.append(id_try)
				continue
		# Try encoding path
		enc = tokenizer(aa, add_special_tokens=False).get("input_ids", [])
		if isinstance(enc, list) and len(enc) == 1 and isinstance(enc[0], int):
			ids.append(enc[0])
			continue
		fail.append(aa)
	if fail:
		raise ValueError(f"Could not determine token ids for amino acids: {fail}")
	# Deduplicate while preserving order
	seen: Dict[int, None] = {}
	uniq: List[int] = []
	for i in ids:
		if i not in seen:
			seen[i] = None
			uniq.append(i)
	return uniq
