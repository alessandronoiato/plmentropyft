from __future__ import annotations

from typing import List, Tuple

import math
import random


def _strip_gaps(seq: str) -> str:
    """Remove gap characters ('-' and spaces) from a sequence."""
    if seq is None:
        return ""
    s = str(seq)
    return "".join(ch for ch in s if ch not in ("-", " "))


def sequence_identity_hamming(a: str, b: str) -> float:
    """Compute ungapped Hamming-like identity between two sequences.

    Identity = matches over aligned positions divided by min(len(a_ungap), len(b_ungap)).
    Aligned positions are indices i < min(len(a), len(b)); positions with gaps are skipped.
    Returns 0.0 if either sequence becomes empty after stripping gaps.
    """
    a0 = a or ""
    b0 = b or ""
    a_ng = _strip_gaps(a0)
    b_ng = _strip_gaps(b0)
    denom = min(len(a_ng), len(b_ng))
    if denom <= 0:
        return 0.0
    limit = min(len(a0), len(b0))
    matches = 0
    used = 0
    for i in range(limit):
        ca = a0[i]
        cb = b0[i]
        if ca in ("-", " ") or cb in ("-", " "):
            continue
        used += 1
        if ca == cb:
            matches += 1
        if used >= denom:
            break
    return float(matches) / float(denom)


def _nw_align_count_matches(a: str, b: str, gap_penalty: int = 1) -> Tuple[int, int]:
    """Needleman–Wunsch global alignment that returns (#matches_in_alignment, denom).

    Scoring: match = +1, mismatch = 0, gap = -gap_penalty.
    After alignment, we count non-gap matches; denominator is min(len(a_ungap), len(b_ungap)).
    """
    a = a or ""
    b = b or ""
    n = len(a)
    m = len(b)
    a_ng_len = len(_strip_gaps(a))
    b_ng_len = len(_strip_gaps(b))
    denom = min(a_ng_len, b_ng_len)
    if denom <= 0:
        return 0, 0

    # DP tables
    # score[i][j] = best score for aligning a[:i] with b[:j]
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]  # 0: diag, 1: up (gap in b), 2: left (gap in a)

    # Initialize
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] - gap_penalty
        trace[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] - gap_penalty
        trace[0][j] = 2

    # Fill
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            s_match = 1 if ai == bj and ai not in ("-", " ") and bj not in ("-", " ") else 0
            diag = score[i - 1][j - 1] + s_match
            up = score[i - 1][j] - gap_penalty
            left = score[i][j - 1] - gap_penalty
            if diag >= up and diag >= left:
                score[i][j] = diag
                trace[i][j] = 0
            elif up >= left:
                score[i][j] = up
                trace[i][j] = 1
            else:
                score[i][j] = left
                trace[i][j] = 2

    # Traceback to count matches
    i = n
    j = m
    matches = 0
    while i > 0 or j > 0:
        t = trace[i][j]
        if t == 0:
            ai = a[i - 1]
            bj = b[j - 1]
            if ai == bj and ai not in ("-", " ") and bj not in ("-", " "):
                matches += 1
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return matches, denom


def sequence_identity_global(a: str, b: str, gap_penalty: int = 1) -> float:
    """Compute Needleman–Wunsch global alignment identity.

    Identity = non-gap matches in optimal global alignment / min(len(a_ungap), len(b_ungap)).
    Returns 0.0 if denominator is 0.
    """
    matches, denom = _nw_align_count_matches(a, b, gap_penalty=gap_penalty)
    if denom <= 0:
        return 0.0
    return float(matches) / float(denom)


def topk_distance_avg(
    sequences: List[str],
    *,
    mode: str = "global",
    topk_percent: int = 5,
    num_pairs: int = 5000,
    seed: int = 123,
    gap_penalty: int = 1,
) -> float:
    """Monte Carlo average of the top k% pairwise distances.

    - Distance = 1 - identity
    - mode: "global" (Needleman–Wunsch) or "hamming" (ungapped)
    - Returns NaN if <2 sequences or no distances available
    """
    seqs = [s for s in sequences if isinstance(s, str) and len(s) > 0]
    if len(seqs) < 2:
        return float("nan")

    rng = random.Random(seed)
    n = len(seqs)
    max_pairs = n * (n - 1) // 2
    sample_with_replacement = num_pairs > max_pairs

    # Precompute unique pairs without replacement if feasible
    pairs: List[Tuple[int, int]] = []
    if not sample_with_replacement:
        # Reservoir-like generation of random unique pairs
        # For simplicity: enumerate all if it's modest; otherwise sample
        if max_pairs <= 200000:
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((i, j))
            rng.shuffle(pairs)
            pairs = pairs[: num_pairs]
        else:
            seen = set()
            while len(pairs) < num_pairs:
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                pairs.append((a, b))
    else:
        for _ in range(num_pairs):
            i = rng.randrange(n)
            j = rng.randrange(n)
            while j == i:
                j = rng.randrange(n)
            a, b = (i, j) if i < j else (j, i)
            pairs.append((a, b))

    dists: List[float] = []
    for i, j in pairs:
        sa = seqs[i]
        sb = seqs[j]
        if mode == "hamming":
            ident = sequence_identity_hamming(sa, sb)
        else:
            ident = sequence_identity_global(sa, sb, gap_penalty=gap_penalty)
        dists.append(max(0.0, min(1.0, 1.0 - ident)))

    if len(dists) == 0:
        return float("nan")
    dists.sort(reverse=True)
    k_count = max(1, int(math.ceil((topk_percent / 100.0) * len(dists))))
    topk = dists[:k_count]
    return float(sum(topk) / len(topk))


