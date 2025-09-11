import argparse
import os
import sys
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.token_utils import get_amino_acid_token_ids, AMINO_ACIDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nferruz/ProtGPT2")
    parser.add_argument("--cache_dir", type=str, default=os.path.join(_PROJECT_ROOT, "hf_cache"))
    parser.add_argument("--local_files_only", action="store_true", default=False)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        tok.padding_side = "left"
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(args.model_id, cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)

    # Extract AA ids via our utility
    aa_ids: List[int] = get_amino_acid_token_ids(tok)
    print(f"Extracted AA ids ({len(aa_ids)}): {aa_ids}")

    # Show mapping per letter and compare to space-prefixed
    print("\nToken mapping per amino acid (letter -> extracted_id | id(letter) | id(' '+letter) | token(extracted_id)):")
    for i, aa in enumerate(AMINO_ACIDS):
        extracted_id = aa_ids[i] if i < len(aa_ids) else None
        id_letter = tok.convert_tokens_to_ids(aa)
        id_space = tok.convert_tokens_to_ids(" " + aa)
        tok_str = tok.convert_ids_to_tokens([extracted_id])[0] if extracted_id is not None else None
        print(f"  {aa}: {extracted_id} | {id_letter} | {id_space} | {tok_str}")

    # Compute BOS next-token distribution and allowed mass
    with torch.no_grad():
        bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        inp = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        attn = torch.ones_like(inp)
        logits = model(input_ids=inp, attention_mask=attn).logits[0, -1]
        logZ_full = torch.logsumexp(logits, dim=-1).item()
        aa_ids_valid = [i for i in aa_ids if isinstance(i, int) and i >= 0 and i < logits.size(0)]
        if len(aa_ids_valid) != 20:
            print(f"\n[WARN] AA id list has {len(aa_ids_valid)} valid ids (expected 20).")
        if aa_ids_valid:
            logZ_aa = torch.logsumexp(logits[aa_ids_valid], dim=-1).item()
            frac = float(torch.exp(torch.tensor(logZ_aa - logZ_full)).item())
        else:
            logZ_aa = float('nan')
            frac = float('nan')
        print(f"\nBOS raw-mass on AA set: logZ_aa={logZ_aa:.6f}, logZ_full={logZ_full:.6f}, frac=exp(diff)={frac:.6e}")

        # Show top-20 next tokens and whether they are in AA set
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=20)
        print("\nTop-20 next tokens at BOS:")
        for rank in range(topk.indices.size(0)):
            tid = int(topk.indices[rank].item())
            p = float(topk.values[rank].item())
            tstr = tok.convert_ids_to_tokens([tid])[0]
            in_aa = tid in aa_ids_valid
            print(f"  {rank+1:2d}. id={tid:5d} p={p: .6e} in_AA={in_aa} token={tstr}")


if __name__ == "__main__":
    main()


