## Algorithm 3 log-likelihood for discrete diffusion LMs: applicability to D3PM and MDLM

### Short answer
- **D3PM**: Not generally applicable. Algorithm 3 only applies to a restricted “mask-absorbing, single-site unmasking” variant. For standard D3PM corruption kernels (parallel, non‑monotone discrete transitions), use the diffusion ELBO rather than exact likelihood via Algorithm 3.
- **MDLM**: Applicable when the model uses a monotone “mask → unmask” reverse process that only fills masked positions (never alters already set tokens) with a known position-selection schedule. Minor bookkeeping is needed if multiple tokens are filled per step.

### What Algorithm 3 is doing (intuition)
It computes the exact data likelihood by summing over all reverse unmasking orders using a dynamic program (DP) on the subset lattice of positions. This works when reverse steps only “add information” and the probability of choosing the next position(s) is known.

### Conditions required for exactness
- **Monotone reverse evolution**: reverse steps only reveal token values; once a token is set, it never changes.
- **Order-invariant conditionals**: each step contributes a factor of the form \(p_\theta(x_i \mid x_S)\) for revealing token \(i\) given the already-revealed set \(S\), independent of the order in which \(S\) was revealed.
- **Known schedule weights**: the probability of picking the next position(s) is known (e.g., uniform over remaining masked positions), making path weights explicit.
- **Known base state**: the chain starts from a known absorbing state (e.g., all-[MASK]) with no latent integrals.

For the common “reveal one token per step, uniform schedule” case:

Let \(V\) be the set of sequence positions and define \(a_i(S) = p_\theta(x_i \mid x_S)\). Then

\[\begin{aligned}
dp(\varnothing) &= 1,\\
dp(T) &= \left(\tfrac{1}{|T|}\right) \sum_{i\in T} dp(T\setminus\{i\})\; a_i\big(T\setminus\{i\}\big),\quad T\neq\varnothing,\\
p_\theta(x) &= dp(V).
\end{aligned}\]

Other schedules replace \(1/|T|\) with the appropriate next-position weight.

### Mapping to D3PM
- **What D3PM is**: a discrete diffusion family where the forward corruption at each time uses a token transition matrix \(Q_t\) applied in parallel across positions; reverse predicts \(p_\theta(x_{t-1}\mid x_t)\). Tokens can jump among the full vocabulary; the chain is not monotone with respect to a “revealed set”.
- **Why Algorithm 3 usually doesn’t apply**: the state space is the full token lattice; multiple positions change in parallel; already-set tokens can change again. This breaks monotonicity and order-invariance. Exact likelihood would require summing over all discrete paths, which is intractable; D3PM therefore uses the variational ELBO.
- **When it can apply**: only if D3PM is restricted to an absorbing-mask forward process and a reverse that unmasks positions one-at-a-time (or exchangeable blocks) without ever changing revealed tokens, with a known next-position policy. That special case reduces to the masked‑unmask setting above.

### Mapping to MDLM
- **What MDLM is**: a masked discrete diffusion LM where the forward process increases the number of [MASK] tokens and the reverse process predicts original tokens at masked sites.
- **Applicability**:
  - **Monotone reverse with known schedule**: Algorithm 3 applies directly. Use the DP above; if the schedule is uniform over remaining masked positions, include the \(1/|T|\) factor; otherwise use the schedule’s weights.
  - **Multiple sites per step**: either expand a \(k\)-site step into the \(k!\) internal single-site orders (include the combinatorial weight), or refine the schedule to single-site steps if within-step choices are exchangeable.
  - **Non‑monotone variants** (e.g., re-masking or revising already-filled tokens): Algorithm 3 is no longer exact; fall back to bounds or path sampling.

### Practical notes and recommendations
- **Complexity**: exact DP is \(\mathcal{O}(L\,2^L)\) time/memory for sequence length \(L\); feasible only for short sequences.
- **For long sequences**:
  - Monte‑Carlo over reveal orders with log‑sum‑exp (importance sampling),
  - Lower bounds (e.g., ELBO/Jensen) or schedule‑aware upper bounds,
  - Block‑wise DP for short spans plus approximate decomposition across spans.
- **Standard D3PM**: use the diffusion ELBO (as in discrete DDPM/D3PM literature), not Algorithm 3.
- **Standard MDLM (mask→unmask)**: Algorithm 3/DP is the right tool for exact likelihood on short sequences or as a basis for MC estimators on long ones.

### Optional implementation sketch (DP over revealed subsets)
```python
# x: observed sequence (length L)
# cond_prob(i, S): model conditional p_theta(x_i | x_S)
# schedule_weight(i, S, T): probability of selecting i next given T (default 1/|T|)

from collections import defaultdict
from itertools import combinations

def all_subsets_of_size(V, k):
    for combo in combinations(V, k):
        yield frozenset(combo)

def llik_dp(x, V, cond_prob, schedule_weight=lambda i, S, T: 1.0/len(T)):
    dp = defaultdict(float)
    dp[frozenset()] = 1.0
    L = len(V)
    for size in range(1, L + 1):
        for T in all_subsets_of_size(V, size):
            s = 0.0
            for i in T:
                S = T - {i}
                s += dp[S] * cond_prob(i, S) * schedule_weight(i, S, T)
            dp[T] = s
    return dp[frozenset(V)]
```

This mirrors the recursion above. Replace the schedule weight if the next‑position policy is non‑uniform.


