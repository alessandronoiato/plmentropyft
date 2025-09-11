import os
import csv
from typing import List

import torch
from transformers.trainer_callback import TrainerCallback

from env.protein_env import ProteinEnv
from .protein_sequence_eval import enumerate_sequence_probs


class ExactEntropyLogger(TrainerCallback):
    def __init__(self, trainer, tokenizer, env: ProteinEnv, aa_ids: List[int], id_eos: int, out_dir: str = "outputs", max_enumeration_horizon: int = 3):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.env = env
        self.aa_ids = aa_ids
        self.id_eos = id_eos
        self.out_dir = out_dir
        self.max_enumeration_horizon = max_enumeration_horizon
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, "grpo_entropy_timeseries.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "entropy_nats"])  # header

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        model_eval = self.trainer.accelerator.unwrap_model(self.trainer.model)
        # Skip enumeration for large horizons (intractable)
        try:
            if getattr(self.env, "config", None) is not None:
                if getattr(self.env.config, "horizon", 9999) > self.max_enumeration_horizon:
                    return
        except Exception:
            pass
        try:
            with torch.no_grad():
                H, _ = enumerate_sequence_probs(model_eval, self.tokenizer, self.env, self.aa_ids, self.id_eos)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([int(state.global_step), float(H)])
        except RuntimeError:
            # Enumeration may blow up for large horizons; skip silently
            pass


