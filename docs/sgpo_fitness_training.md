# SGPO Fitness Training with Entropy Regularization

This document describes the `run_sgpo_fitness.py` script, which fine-tunes a protein language model (ProGen2) to generate high-fitness protein sequences using GRPO (Group Relative Policy Optimization) with entropy regularization.

## Objective

We optimize a protein language model π to generate sequences with high fitness scores from an external oracle, while maintaining diversity through entropy regularization and stability through KL divergence anchoring.

### Mathematical Formulation

We maximize the objective:

$$
J(\pi) = \mathbb{E}_{\pi}[\lambda \cdot r(y)] + \mu \cdot H(\pi) - \eta \cdot \text{KL}(\pi \| \pi_{\text{base}})
$$

Where:
- **r(y)**: Fitness score from the SGPO oracle (TrpB enzyme activity)
- **λ (fitness_scale)**: Weight on the external fitness reward
- **μ (entropy_coef)**: Entropy bonus coefficient (encourages diversity)
- **η (first_variation_coef)**: KL penalty to base model (prevents drift from pretrained knowledge)
- **π_base**: The original pretrained/fine-tuned ProGen2 model

### First Variation (Per-Sequence Reward)

GRPO optimizes using per-sequence rewards. The first variation of the objective gives us the reward function:

$$
R(y) = \lambda \cdot r(y) - (\mu + \eta) \cdot \log p_{\text{ref}}(y) + \eta \cdot \log p_{\text{base}}(y)
$$

Where:
- **p_ref**: The reference policy (frozen copy of the current policy)
- **p_base**: The base model (optional, for anchoring)

**Interpretation:**
- The term `−μ · log p_ref(y)` rewards sequences that are **surprising** under the reference model (entropy bonus)
- The term `−η · log p_ref(y) + η · log p_base(y)` penalizes deviation from the base model (KL penalty)
- Combined: `−(μ + η) · log p_ref(y) + η · log p_base(y)`

## Components

### 1. ProGen2 Model
- **Source**: SGPO repository's fine-tuned ProGen2 (`checkpoints/causalLM_finetune/TrpB/best`)
- **Architecture**: GPT-2 style transformer trained on protein sequences
- **Fine-tuning**: Pre-fine-tuned on TrpB MSA data by SGPO authors

### 2. SGPO Fitness Oracle
- **Type**: Ensemble of 5 MLPs
- **Input**: 15-character "Combo" sequence (amino acids at mutated positions)
- **Output**: Predicted fitness score (normalized, ~0-1 range)
- **Training data**: Experimental TrpB enzyme activity measurements

### 3. Sequence Processing Pipeline

```
Generated Sequence (full length, ~390 AA)
         ↓
    MAFFT Alignment (to parent TrpB)
         ↓
    Projection (extract 15 mutated positions)
         ↓
    Combo Sequence (15 chars)
         ↓
    Oracle Scoring → Fitness
```

### 4. GRPO Trainer
- **Framework**: TRL (Transformer Reinforcement Learning)
- **Algorithm**: Group Relative Policy Optimization
- **β parameter**: Additional KL penalty in GRPO's loss (separate from our η term)

## Usage

### Explore Mode
Analyze the fitness landscape and dataset:
```bash
python scripts/run_sgpo_fitness.py --mode explore \
    --sgpo_repo ~/Code/SGPO
```

### Train Mode
Fine-tune the model with fitness + entropy regularization:
```bash
python scripts/run_sgpo_fitness.py --mode train \
    --sgpo_repo ~/Code/SGPO \
    --steps 100 \
    --batch_size 32 \
    --fitness_scale 1.0 \
    --entropy_coef 0.1 \
    --first_variation_coef 0.0 \
    --beta 0.01 \
    --learning_rate 1e-5
```

### Eval Mode
Evaluate a trained model:
```bash
python scripts/run_sgpo_fitness.py --mode eval \
    --sgpo_repo ~/Code/SGPO \
    --progen2_path outputs/sgpo/final_model \
    --eval_samples 500 \
    --out_dir outputs/sgpo_after_grpo
```

## Key Parameters

| Parameter | Symbol | Description | Default |
|-----------|--------|-------------|---------|
| `--fitness_scale` | λ | Weight on fitness reward | 1.0 |
| `--entropy_coef` | μ | Entropy bonus (diversity) | 1.0 |
| `--first_variation_coef` | η | KL penalty to base model | 0.0 |
| `--beta` | β | GRPO's built-in KL penalty | 0.01 |
| `--learning_rate` | - | Adam learning rate | 1e-5 |
| `--steps` | - | Number of training steps | 100 |
| `--batch_size` | - | Batch size per device | 32 |
| `--num_generations` | - | Generations per prompt | 16 |

## Outputs

### Training Outputs (`outputs/sgpo/`)
- `trainer_output/checkpoint-*/`: Model checkpoints
- `final_model/`: Final trained model weights
- `fitness_log.json`: Per-batch fitness statistics
- `grpo_approx_kl_in_update.csv`: KL divergence between policy and reference
- `grpo_first_variation_in_update.csv`: First variation term statistics

### Evaluation Outputs (`outputs/sgpo_after_grpo/`)
- `eval_results.json`: Full evaluation results including:
  - Generated sequences and their fitness scores
  - Diversity metrics (Shannon entropy, Hamming distance, unique count)
  - Comparison with SGPO paper metrics

## Preventing Mode Collapse

Mode collapse occurs when the model finds a single high-fitness sequence and repeatedly generates it. We prevent this through:

1. **Entropy bonus (μ > 0)**: Rewards sequences that are surprising under the reference model
2. **KL penalty (β > 0)**: GRPO's built-in penalty for deviating from reference
3. **First variation term (η > 0)**: Anchors policy to the pretrained base model

**Recommended settings for balanced optimization:**
```bash
--entropy_coef 0.1 --beta 0.01
```

## Comparison with SGPO Paper

SGPO uses a different optimization approach (Bayesian optimization over latent space). Our approach:
- Uses the same fitness oracle
- Uses the same fine-tuned ProGen2 as the starting point
- Applies direct policy gradient optimization (GRPO) instead of BO
- Adds explicit entropy regularization for diversity

Evaluation metrics match SGPO's `analysis.ipynb`:
- Mean/Max/Q90 Fitness
- Shannon Entropy of position distributions
- Pairwise Hamming Distance
- Unique sequence count

