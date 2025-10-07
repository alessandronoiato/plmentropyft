from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import csv


def write_sequence_probs_csv(path: str, seqs: List[Tuple[str, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "probability"])  # header
        for s, p in seqs:
            w.writerow([s, float(p)])


def write_validity_csv(path: str, validity_mode: str, per_sample_records: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if validity_mode == "esmfold":
            w.writerow(["sequence", "valid", "fold_ok", "plddt_mean", "plddt_median", "fold_error"])  # header
            for rec in per_sample_records:
                w.writerow([
                    rec.get("sequence"),
                    int(rec.get("valid", 0)),
                    bool(rec.get("fold_ok", False)),
                    rec.get("plddt_mean"),
                    rec.get("plddt_median"),
                    rec.get("fold_error"),
                ])
        else:
            w.writerow(["sequence", "valid"])  # header
            for rec in per_sample_records:
                w.writerow([rec.get("sequence"), int(rec.get("valid", 0))])


def build_report(
    *,
    horizon: int,
    seqs_before: List[Tuple[str, float]],
    seqs_after: List[Tuple[str, float]],
    H_before: float,
    H_after: float,
    H_before_valid: float,
    H_after_valid: float,
    V_before: float,
    V_after: float,
    Ltok_before: float,
    Ltok_after: float,
    Lres_before: float,
    Lres_after: float,
    diversity_metric_used: str,
    vendi_before: Optional[float] = None,
    vendi_after: Optional[float] = None,
    vendi_sigma_used_before: Optional[float] = None,
    vendi_sigma_used_after: Optional[float] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "horizon": int(horizon),
        "num_sequences_before": int(len(seqs_before)),
        "num_sequences_after": int(len(seqs_after)),
        "before_entropy_nats": float(H_before) if H_before == H_before else float("nan"),
        "after_entropy_nats": float(H_after) if H_after == H_after else float("nan"),
        "before_entropy_nats_valid_only": float(H_before_valid) if H_before_valid == H_before_valid else float("nan"),
        "after_entropy_nats_valid_only": float(H_after_valid) if H_after_valid == H_after_valid else float("nan"),
        "before_mean_validity": float(V_before),
        "after_mean_validity": float(V_after),
        "before_mean_token_length_to_eos": float(Ltok_before),
        "after_mean_token_length_to_eos": float(Ltok_after),
        "before_mean_residue_length_to_eos": float(Lres_before),
        "after_mean_residue_length_to_eos": float(Lres_after),
        "mean_token_length_delta": float(Ltok_after - Ltok_before),
        "mean_residue_length_delta": float(Lres_after - Lres_before),
        "sum_probs_before": 1.0,
        "sum_probs_after": 1.0,
        "diversity_metric_used": diversity_metric_used,
    }
    if vendi_before is not None:
        report["before_diversity"] = float(vendi_before)
    if vendi_after is not None:
        report["after_diversity"] = float(vendi_after)
    if vendi_sigma_used_before is not None:
        report["before_vendi_sigma_used"] = vendi_sigma_used_before
    if vendi_sigma_used_after is not None:
        report["after_vendi_sigma_used"] = vendi_sigma_used_after
    return report


