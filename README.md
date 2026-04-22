# plmentropyft

Fine-tuning protein language models with **GRPO** (Group Relative Policy Optimization) using an entropy-based reward derived from self-surprise and first-variation objectives.

## Overview

This repository implements reinforcement learning-style fine-tuning of protein sequence models (e.g. [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2)) via Hugging Face [TRL](https://github.com/huggingface/trl). The reward encourages the policy to generate sequences that are both surprising under a reference model and constrained relative to a frozen base model:

$$R = -(\mu + \eta)\log p_{\text{ref}}(\text{seq}) + \eta \log p_{\text{base}}(\text{seq})$$

where `entropy_coef` (μ) controls self-surprise and `first_variation_coef` (η) controls the base-model anchor.

A secondary pipeline integrates with [SGPO](https://github.com/jsunn-y/SGPO) for fitness-oracle RL on the TrpB protein.

## Features

- **GRPO training** for causal protein LMs with custom reward shaping
- **Protein-specific environments** with amino-acid legality masks and horizon control
- **ESMFold validity** checking for generated sequences
- **Vendi diversity score** using ESM2 embeddings
- **Quantile Hausdorff distance** for sequence-set diversity measurement
- **SGPO integration** for TrpB fitness: explore / train / eval / Pareto sweep modes
- **WandB logging** (optional)

## Installation

```bash
git clone https://github.com/<you>/plmentropyft
cd plmentropyft
pip install -r requirements.txt
```

For GPU + folding support (CUDA 12.1, OpenFold, Facebook ESM):

```bash
pip install -r requirements-cuda.txt
```

For Pareto comparison plots:

```bash
pip install matplotlib pandas
```

## Usage

### GRPO fine-tuning

```bash
python scripts/run_prot_trl.py \
  --model_id nferruz/ProtGPT2 \
  --steps 64 \
  --batch_size 16 \
  --entropy_coef 1.0 \
  --first_variation_coef 0.5
```

Key flags: `--horizon` (sequence length), `--wandb_project` (optional WandB logging), `--eval_esm_fold` (validity), `--eval_vendi` (diversity).

### SGPO / TrpB fitness pipeline

```bash
# Explore: generate sequences and score with fitness oracle
python scripts/run_sgpo_fitness.py --mode explore --sgpo_repo ~/path/to/SGPO

# Train: GRPO with fitness reward
python scripts/run_sgpo_fitness.py --mode train --sgpo_repo ~/path/to/SGPO

# Pareto sweep
python scripts/run_sgpo_fitness.py --mode pareto --sgpo_repo ~/path/to/SGPO
```

### Compare Pareto fronts

```bash
python scripts/compare_pareto.py --grpo_csv outputs/pareto.csv --dpo_csv ~/SGPO/results/dpo.csv
```

## Project Structure

```
scripts/
  run_prot_trl.py          # Main GRPO training & evaluation
  run_sgpo_fitness.py      # TrpB fitness pipeline (SGPO integration)
  compare_pareto.py        # DPO vs GRPO Pareto comparison plots
env/
  protein_env.py           # Amino-acid legality mask environment
  protein_piece_env.py     # Piece-level horizon environment
utils/
  protein_reward.py        # GRPO reward: entropy + first-variation terms
  sgpo/                    # SGPO integration (ProGen2, MAFFT, oracle, pipeline)
  vendi.py                 # Vendi diversity score
  fold_validity.py         # ESMFold-based validity checking
  sequence_distance.py     # Quantile Hausdorff distance
  wandb_logger.py          # Optional WandB integration
```

## Dependencies

| Package | Role |
|---------|------|
| `torch`, `transformers`, `accelerate` | Core ML |
| `trl==0.22.0` | GRPO trainer |
| `peft` | Parameter-efficient fine-tuning |
| `fair-esm` | ESM2 embeddings & ESMFold |
| `biopython` | Sequence utilities |
| `omegaconf` | Configuration |
| `wandb` | Experiment tracking (optional) |

## Citation

If you use the entropy / first-variation reward formulation, please cite the relevant TRL and ESM papers alongside this repository.
