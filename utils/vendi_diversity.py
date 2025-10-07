from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math
import warnings

import numpy as np
import torch


def _sanitize_sequence_for_esm(seq: str) -> str:
    """Sanitize an input protein sequence for ESM embedding.

    - Upper-case
    - Drop non-canonical characters like 'Ċ'
    - Map anything not in the 20 AA alphabet to 'X'
    """
    if seq is None:
        return ""
    s = str(seq).strip().upper()
    # Remove 'Ċ' tokens entirely (observed in some GPT tokenizations)
    s = s.replace("Ċ", "")
    aa20 = set("ACDEFGHIKLMNPQRSTVWY")
    out_chars: List[str] = []
    for ch in s:
        out_chars.append(ch if ch in aa20 else "X")
    return "".join(out_chars)


def load_esm2(
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    dtype: str = "float32",
):
    """Load an ESM2 model and its alphabet from fair-esm.

    Returns (model, alphabet, batch_converter). Raises ImportError if esm is missing.
    """
    try:
        import esm  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        raise ImportError(
            "ESM library not found. Please install fair-esm (pip install fair-esm)."
        ) from e

    # Resolve the pretrained loader function dynamically by name
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"Unknown ESM2 model name: {model_name}")
    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()

    model.eval()
    model_device = torch.device(device)
    model.to(model_device)

    # dtype handling
    if str(dtype).lower() in ("float16", "fp16", "half"):
        model.half()
    elif str(dtype).lower() in ("bfloat16", "bf16"):
        model.bfloat16()
    # else keep float32

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


@torch.no_grad()
def esm2_embed_sequences(
    sequences: List[str],
    *,
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 16,
    layer: Optional[int] = None,
    l2_normalize: bool = False,
) -> torch.Tensor:
    """Embed sequences with ESM2 and return a tensor of shape [N, D].

    Pooling: mean over per-token representations excluding special tokens.
    """
    if len(sequences) == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    model, alphabet, batch_converter = load_esm2(model_name=model_name, device=device, dtype=dtype)

    sanitized = [_sanitize_sequence_for_esm(s) for s in sequences]
    # Replace empty strings with a minimal placeholder to avoid crashes; will embed but be filtered downstream
    sanitized = [s if len(s) > 0 else "X" for s in sanitized]

    model_device = next(model.parameters()).device
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(str(dtype).lower(), torch.float32)

    reps: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(sanitized), batch_size):
        end = min(len(sanitized), start + batch_size)
        batch_labels = [f"seq_{i}" for i in range(start, end)]
        # ESM expects a list of (label, sequence) pairs
        data = list(zip(batch_labels, sanitized[start:end]))
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(model_device)
        # forward with representation layers output
        out = model(tokens, repr_layers=[layer] if layer is not None else [33], return_contacts=False)
        # Default to last layer if not set; many ESM2 variants use 33 or 36 layers
        last_layer = layer if layer is not None else max(out["representations"].keys())
        token_reps = out["representations"][last_layer]  # [B, T, D]

        # Mean pool over tokens excluding BOS/EOS per ESM convention (alphabet.cls_idx and alphabet.eos_idx)
        bos_idx = alphabet.cls_idx if hasattr(alphabet, "cls_idx") else 0
        eos_idx = alphabet.eos_idx if hasattr(alphabet, "eos_idx") else None
        # Build masks: exclude BOS and EOS if present
        mask = torch.ones(token_reps.size()[:2], dtype=torch.bool, device=token_reps.device)
        mask[:, bos_idx] = False
        if eos_idx is not None and eos_idx < mask.size(1):
            mask[:, eos_idx] = False
        mask = mask.unsqueeze(-1)
        token_reps = token_reps.masked_fill(~mask, 0.0)
        lengths = mask.sum(dim=1).clamp_min(1)
        pooled = token_reps.sum(dim=1) / lengths
        pooled = pooled.to(dtype=torch_dtype)
        reps.append(pooled)

    emb = torch.cat(reps, dim=0)  # [N, D]
    if l2_normalize and emb.numel() > 0:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb


def _pairwise_squared_distances(x: torch.Tensor) -> torch.Tensor:
    # x: [N, D]
    x_norm = (x * x).sum(dim=1, keepdim=True)
    d2 = x_norm + x_norm.T - 2.0 * (x @ x.T)
    d2 = torch.clamp(d2, min=0.0)
    return d2


def vendi_from_embeddings(
    embeddings: torch.Tensor,
    *,
    weights: Optional[List[float]] = None,
    kernel: str = "cosine",
    sigma: Optional[float] = None,
) -> Dict[str, float]:
    """Compute Vendi score from embeddings.

    - If weights are provided, they should sum to 1.0 (we will renormalize defensively).
    - Returns a dict with keys: vendi_score, shannon_entropy_nats, sigma_used, debug.
    """
    n, d = (int(embeddings.size(0)), int(embeddings.size(1))) if embeddings.numel() > 0 else (0, 0)
    if n == 0:
        return {"vendi_score": float("nan"), "shannon_entropy_nats": float("nan"), "sigma_used": None, "debug": {"n": 0}}

    x = embeddings.to(dtype=torch.float64)
    # Build kernel matrix K in float64 for stability
    if kernel == "cosine":
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        K = (x @ x.T).clamp(min=-1.0, max=1.0)
        sigma_used = None
    elif kernel == "rbf":
        d2 = _pairwise_squared_distances(x)
        if sigma is None:
            # median heuristic over upper triangle (excluding diagonal)
            with torch.no_grad():
                tri = d2.cpu().numpy()
                iu = np.triu_indices(n, k=1)
                vals = tri[iu]
                med = float(np.median(vals)) if vals.size > 0 else 1.0
            sigma_used = math.sqrt(max(med, 1e-12))
        else:
            sigma_used = float(sigma)
        denom = 2.0 * (sigma_used ** 2)
        K = torch.exp(-d2 / max(denom, 1e-12))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Apply weights: M = sqrt(W) K sqrt(W)
    if weights is not None and len(weights) == n:
        # Ensure weights live on the same device as embeddings/kernel
        w = torch.tensor(weights, dtype=torch.float64, device=x.device)
        w = w / (w.sum() + 1e-40)
        Wsqrt = torch.diag(torch.sqrt(torch.clamp(w, min=0.0)))
        M = Wsqrt @ K @ Wsqrt
    else:
        M = K
        w = None

    # Eigenvalues
    eigvals = torch.linalg.eigvalsh(M)
    eigvals = torch.clamp(eigvals.real, min=0.0)
    trace = eigvals.sum()
    if not torch.isfinite(trace) or trace.item() <= 0.0:
        return {
            "vendi_score": float("nan"),
            "shannon_entropy_nats": float("nan"),
            "sigma_used": sigma_used if kernel == "rbf" else None,
            "debug": {"trace": float(trace.item()) if torch.isfinite(trace) else float("nan"), "n": n},
        }
    p = eigvals / trace
    # Shannon entropy in nats
    with torch.no_grad():
        p_np = p.cpu().numpy()
        p_np = np.clip(p_np, 1e-40, 1.0)
        H = float(-np.sum(p_np * np.log(p_np)))
    vendi = math.exp(H)
    return {
        "vendi_score": float(vendi),
        "shannon_entropy_nats": float(H),
        "sigma_used": sigma_used if kernel == "rbf" else None,
        "debug": {"n": n, "d": d, "trace": float(trace.item()), "min_eig": float(eigvals.min().item())},
    }


def vendi_from_sequences(
    sequences: List[str],
    *,
    weights: Optional[List[float]] = None,
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 16,
    kernel: str = "cosine",
    sigma: Optional[float] = None,
) -> Dict[str, float]:
    """End-to-end Vendi from raw sequences.

    Returns a dict with vendi_score, shannon_entropy_nats, sigma_used and debug info.
    """
    if len(sequences) == 0:
        return {"vendi_score": float("nan"), "shannon_entropy_nats": float("nan"), "sigma_used": None, "debug": {"n": 0}}

    emb = esm2_embed_sequences(
        sequences,
        model_name=model_name,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        layer=None,
        l2_normalize=(kernel == "cosine"),
    )
    return vendi_from_embeddings(embeddings=emb, weights=weights, kernel=kernel, sigma=sigma)


