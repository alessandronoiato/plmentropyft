## Detailed approaches to demonstrate post‑finetuning access to new modes

This note expands each idea in `mode_discovery.md` with what to compute, how, and how to interpret results.

### 1) Novel‑valid rate under the pre‑model
- Goal: quantify how often the post model produces valid sequences that are rare under the pre model.
- Inputs:
  - A set of pre‑generated sequences and a set of post‑generated sequences (equal budget if possible).
  - Log‑likelihoods −log p_pre(seq) evaluated under the pre model.
  - Validity labels (e.g., ESMFold mean pLDDT ≥ threshold).
- Method:
  1) Compute the empirical 95th percentile τ of −log p_pre over the pre sample.
  2) Define “pre‑tail” = {seq: −log p_pre(seq) ≥ τ}.
  3) Report R_τ(pre) = P_pre[valid ∧ pre‑tail] and R_τ(post) = P_post[valid ∧ pre‑tail].
- Interpretation: A higher R_τ(post) shows the post model reaches valid regions that lie in the pre model’s low‑probability tail.
- Caveats: choose τ (e.g., 90–99th percentile) to balance stability and rarity; ensure comparable evaluation budgets.

### 2) Tail‑likelihood profile (pre NLL of post valids)
- Goal: check how extreme post valids are under the pre likelihood.
- Inputs: post valid sequences; −log p_pre(seq) and the reference pre distribution.
- Method: map each post valid to its percentile in the pre −log p distribution; plot the CDF or histogram.
- Interpretation: mass in the extreme right tail (e.g., ≥95th percentile) signals “almost inaccessible” under pre.
- Caveats: smooth with sufficient sample sizes; consider reporting the median/90th/95th percentiles.

### 3) Nearest‑neighbor novelty (embedding space)
- Goal: measure how far post valids lie from the pre valid manifold in a semantic space (e.g., ESM2 embeddings).
- Inputs: ESM2 embeddings for pre valids and post valids; a distance metric (cosine/Euclidean).
- Method:
  1) For each post valid embedding, compute its distance to the nearest pre valid embedding.
  2) Summarize with a CDF, median, 90th/95th percentiles, and fraction above a high threshold.
- Interpretation: larger NN distances indicate the post model discovers new neighborhoods of valid sequences.
- Caveats: normalize embeddings consistently; set a distance threshold based on pre intra‑cohort distances.

### 4) Cross‑cohort distances only
- Goal: compare separation between sets without intra‑set effects.
- Inputs: pre valid set V_pre and post valid set V_post; a distance metric (e.g., Hamming or global alignment).
- Method: compute distances only across pairs in V_pre × V_post; summarize top‑k% or full distribution.
- Interpretation: larger top‑k cross distances indicate the post valid set lies farther from pre valids.
- Caveats: computational cost is |V_pre|×|V_post|; use sampling if sets are large.

### 5) Mode discovery via clustering
- Goal: show that post discovers additional (or different) valid clusters.
- Inputs: embeddings of V_pre ∪ V_post; a density clustering method (DBSCAN/HDBSCAN) or k‑means.
- Method:
  1) Cluster the union with fixed hyperparameters.
  2) Plot number of clusters discovered vs evaluation budget for pre and post.
  3) Alternatively, train clusters on pre only, then assign post valids and count assignments to new/tiny clusters.
- Interpretation: more clusters found (or more mass in small/rare clusters) implies new mode discovery.
- Caveats: clustering depends on scale/parameters; report sensitivity analyses.

### 6) Low‑probability bin analysis
- Goal: assess if post attains higher valid rates in regions where pre assigns low probability.
- Inputs: −log p_pre for both cohorts; validity labels.
- Method:
  1) Bin sequences by −log p_pre deciles (using pre’s distribution for bin edges).
  2) For each bin, compute valid rates for pre and post.
  3) Visualize valid‑rate vs bin index for both.
- Interpretation: a larger gap in higher‑NLL bins indicates post finds valid sequences where pre is weak.
- Caveats: ensure enough samples per bin; merge tail bins if sparse.

### 7) Importance‑weight view (rarity under pre)
- Goal: quantify rarity of post valids with importance weights from pre.
- Inputs: post valid sequences; −log p_pre.
- Method: set weights w_i ∝ exp(−NLL_pre(seq_i)) (renormalize). Report weight histograms and effective sample size (ESS = (∑w)^2 / ∑w^2).
- Interpretation: a heavy‑tailed weight distribution (low ESS) indicates many post valids are rare under pre.
- Caveats: normalize weights to avoid overflow; compare with pre’s own weight profile as baseline.

### 8) Quality‑novelty Pareto
- Goal: show simultaneous gains in quality and novelty.
- Inputs: a quality metric (e.g., pLDDT or valid indicator) and a novelty metric (−log p_pre or NN distance).
- Method: scatter points for both cohorts; overlay Pareto frontiers or density contours; compare shifts.
- Interpretation: a shift toward higher quality and higher novelty suggests new, desirable modes.
- Caveats: for fair comparison, use equal budgets; annotate medians/quantiles.

### 9) Equal‑budget, matched‑N curves (variance‑controlled)
- Goal: compare diversity fairly when valid counts differ.
- Inputs: V_pre and V_post, with differing sizes.
- Method:
  1) For a fixed evaluation budget, subsample the larger valid set to match the smaller N.
  2) Compute top‑k distances without replacement for both (use all unique pairs or the same fixed number).
  3) Plot distance vs budget and compare curves.
- Interpretation: post‑above‑pre indicates greater dispersion among valids at equal N (variance‑controlled).
- Caveats: subsampling adds randomness; bootstrap to add error bars.

---

### Practical guidance
- Always report budgets (sequences folded/evaluated) and valid counts for both cohorts.
- Prefer equal budgets and no‑replacement pairing to avoid variance artifacts.
- When using esmfold filtering, cache folds and report thresholds; consider multiple thresholds (e.g., 70/80).
- Combine several views (likelihood tail, embedding novelty, cross‑distances, clustering) for a robust story.


