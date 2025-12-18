# SGPO Fitness Oracle Integration Plan

## Overview

This document outlines the plan to integrate the fitness oracle from [SGPO](https://github.com/jsunn-y/SGPO) into our entropy-regularized fine-tuning framework.

**Target Dataset:** TrpB (Tryptophan Synthase Beta Subunit)

## Objective

Optimize the combined objective:
```
J(π) = E_π[r(y)] + μ H(π) - η KL(π || π_base)
```

with reward per sequence (first variation):
```
R(y) = λ·r(y) - (μ + η)·log π_ref(y) + η·log π_base(y)
```

where:
- `r(y)` = fitness score from SGPO oracle
- `λ` = fitness scale factor
- `μ` = entropy coefficient  
- `η` = KL-to-base coefficient

---

## SGPO Repository Structure

```
SGPO/
├── data/
│   ├── TrpB/           # TrpB fitness data
│   ├── CreiLOV/        # CreiLOV fitness data
│   └── GB1/            # GB1 fitness data
├── oracle/
│   ├── checkpoints/    # Pretrained fitness oracles
│   └── train_oracle.py # Oracle training script
├── models/             # Diffusion model architectures
├── sampling/           # Guided sampling algorithms
└── dataset/            # Dataset objects for training
```

---

## TrpB Dataset

### Background
- **Protein:** Tryptophan synthase β-subunit from *Pyrococcus furiosus*
- **Source:** Wu et al. (2019) combinatorial mutagenesis study
- **Fitness metric:** Catalytic activity (normalized)

### Key Questions to Explore

1. **Sequence format:**
   - Full-length sequences or just mutated positions?
   - Wild-type sequence for reference?
   - Alignment requirements?

2. **Fitness score range:**
   - Raw values vs normalized [0, 1]?
   - Z-scored?
   - Log-transformed?

3. **Dataset size:**
   - Training set size?
   - Validation/test splits?

---

## Oracle Architecture

### Questions to Explore

1. **Model type:**
   - CNN over one-hot encoded sequences?
   - MLP?
   - ESM embeddings + MLP?

2. **Input encoding:**
   - One-hot (20 amino acids)?
   - Positional encoding?
   - Only mutated positions?

3. **Output:**
   - Single fitness score?
   - Uncertainty estimate?

---

## Integration Steps

### Phase 1: Data Exploration
- [ ] Clone SGPO repo or download data from HuggingFace
- [ ] Load TrpB fitness data
- [ ] Analyze fitness score distribution
- [ ] Understand sequence format

### Phase 2: Oracle Integration
- [ ] Understand oracle checkpoint format
- [ ] Implement oracle loading in `SGPOFitnessOracle`
- [ ] Implement sequence encoding
- [ ] Test oracle inference

### Phase 3: Reward Function
- [ ] Integrate fitness scores into reward
- [ ] Add log-probability terms for entropy/KL
- [ ] Balance fitness scale with entropy terms

### Phase 4: Training
- [ ] Run GRPO with fitness-augmented reward
- [ ] Tune hyperparameters (λ, μ, η)
- [ ] Compare with baseline (no fitness)

---

## Fitness Scale Discussion

The key challenge is balancing the fitness term with the entropy/KL terms:

### Option A: Raw Fitness
```
R(y) = r(y) - (μ + η)·log π_ref(y) + η·log π_base(y)
```
- Fitness in [0, 1] vs log-probs in [-100, 0] range
- May need large `λ` to balance

### Option B: Scaled Fitness
```
R(y) = λ·r(y) - (μ + η)·log π_ref(y) + η·log π_base(y)
```
- Tune `λ` to match magnitude of log-prob terms
- E.g., if mean log-prob ≈ -50, use λ ≈ 50

### Option C: Normalized Fitness
```
R(y) = λ·(r(y) - r_mean) / r_std - ...
```
- Z-score fitness to have similar scale as other terms
- More stable across datasets

### Recommendation
Start with **Option B** (scaled raw fitness) and tune `λ` empirically:
1. Compute mean absolute log-prob: `|E[log π_ref]|`
2. Set `λ ≈ |E[log π_ref]| / (r_max - r_min)`
3. This makes fitness and entropy terms comparable

---

## Commands

### Explore Data
```bash
python scripts/run_sgpo_fitness.py \
    --mode explore \
    --dataset TrpB \
    --fitness_data /path/to/SGPO/data/TrpB/fitness.csv \
    --out_dir outputs/sgpo
```

### Train (once implemented)
```bash
python scripts/run_sgpo_fitness.py \
    --mode train \
    --model_id nferruz/ProtGPT2 \
    --fitness_scale 50.0 \
    --entropy_coef 1.0 \
    --first_variation_coef 0.5 \
    --oracle_checkpoint /path/to/SGPO/oracle/checkpoints/TrpB/
```

---

## Next Steps

1. **Clone SGPO repo** (or get data from HuggingFace)
2. **Run explore mode** to understand data format
3. **Review oracle code** in SGPO to understand architecture
4. **Implement oracle loading** based on their format
5. **Discuss fitness scaling** strategy based on data analysis

---

## References

- SGPO Repository: https://github.com/jsunn-y/SGPO
- Paper: "Steering Generative Models with Experimental Data for Protein Fitness Optimization"
- TrpB Dataset: Wu et al. (2019) - Nature Communications

