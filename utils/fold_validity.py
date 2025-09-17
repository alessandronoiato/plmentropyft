import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple


_ESM_MODEL_SINGLETON = None
_ESM_MODEL_DEVICE = None


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_esmfold(device: str = "cpu", dtype: str = "float32"):
    """Lazily load ESMFold model and cache a singleton.

    Parameters
    - device: "cpu" | "cuda"
    - dtype: "float32" | "float16"
    """
    global _ESM_MODEL_SINGLETON, _ESM_MODEL_DEVICE
    try:
        import torch  # noqa: F401
        from esm.pretrained import esmfold_v1
    except Exception as exc:  # pragma: no cover - dependency may be optional
        raise RuntimeError(
            "ESMFold is not available. Please install 'esm' (pip install fair-esm) to use esmfold validity."
        ) from exc

    if _ESM_MODEL_SINGLETON is not None and _ESM_MODEL_DEVICE == device:
        return _ESM_MODEL_SINGLETON

    model = esmfold_v1()
    model = model.eval()
    if device == "cuda":
        model = model.cuda()
    # ESMFold uses float32 by default; allow optional casting
    if dtype == "float16":
        try:
            model = model.half()
        except Exception:
            pass
    _ESM_MODEL_SINGLETON = model
    _ESM_MODEL_DEVICE = device
    return model


def _parse_plddt_from_pdb(pdb_str: str) -> List[float]:
    """Extract per-residue pLDDT from PDB B-factor column.

    ESMFold encodes pLDDT in the B-factor field (0-100).
    """
    plddts: List[float] = []
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM"):
            continue
        # Columns per PDB format: temperature factor at 61-66 (1-based). Python slices are 0-based.
        try:
            tf_str = line[60:66].strip()
            if tf_str:
                plddts.append(float(tf_str))
        except Exception:
            continue
    return plddts


def fold_plddt_stats(
    seqs: List[str],
    *,
    device: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 1,
    cache_dir: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Compute pLDDT stats for sequences using ESMFold.

    Returns list of dicts per input with keys:
    - ok: bool
    - mean_plddt: float | None
    - median_plddt: float | None
    - length: int
    - error: str | None
    """
    import time
    import statistics as stats

    # Prepare cache
    cache_path = None
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "esmfold_plddt_cache.jsonl")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        cache[rec["key"]] = rec
                    except Exception:
                        continue

    model = load_esmfold(device=device, dtype=dtype)

    results: List[Dict[str, Any]] = []
    i = 0
    while i < len(seqs):
        chunk = seqs[i : i + max(1, batch_size)]
        for s in chunk:
            key = _sha1(s)
            cached = cache.get(key) if cache else None
            if cached is not None:
                results.append(
                    {
                        "ok": bool(cached.get("ok", False)),
                        "mean_plddt": cached.get("mean_plddt"),
                        "median_plddt": cached.get("median_plddt"),
                        "length": int(cached.get("length", len(s))),
                        "error": cached.get("error"),
                    }
                )
                continue

            start = time.time()
            try:
                # ESMFold API: infer_pdb returns a PDB string with pLDDT in B-factors
                pdb_str = model.infer_pdb(s)
                plddt = _parse_plddt_from_pdb(pdb_str)
                if len(plddt) == 0:
                    raise RuntimeError("empty pLDDT from PDB")
                mean_val = float(sum(plddt) / len(plddt))
                median_val = float(stats.median(plddt))
                rec = {
                    "ok": True,
                    "mean_plddt": mean_val,
                    "median_plddt": median_val,
                    "length": len(s),
                    "error": None,
                }
            except Exception as exc:  # pragma: no cover - robustness
                rec = {
                    "ok": False,
                    "mean_plddt": None,
                    "median_plddt": None,
                    "length": len(s),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            # Timeout handling
            if timeout_s is not None and (time.time() - start) > timeout_s:
                rec = {
                    "ok": False,
                    "mean_plddt": None,
                    "median_plddt": None,
                    "length": len(s),
                    "error": "timeout",
                }
            results.append(rec)

            # Persist to cache
            if cache_path is not None:
                try:
                    with open(cache_path, "a") as f:
                        out = {
                            "key": key,
                            "ok": rec["ok"],
                            "mean_plddt": rec["mean_plddt"],
                            "median_plddt": rec["median_plddt"],
                            "length": rec["length"],
                            "error": rec["error"],
                        }
                        f.write(json.dumps(out) + "\n")
                except Exception:
                    pass
        i += max(1, batch_size)
    return results


def is_valid_esmfold(
    seq: str,
    *,
    threshold: float = 70.0,
    device: str = "cpu",
    dtype: str = "float32",
) -> Tuple[int, Dict[str, Any]]:
    """Return (0/1, stats) where stats includes pLDDT metrics."""
    stats_list = fold_plddt_stats([seq], device=device, dtype=dtype, batch_size=1)
    st = stats_list[0]
    ok = st.get("ok", False)
    mean_plddt = st.get("mean_plddt")
    valid = 1 if ok and (mean_plddt is not None) and (mean_plddt >= threshold) else 0
    return valid, st


