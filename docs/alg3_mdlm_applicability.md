## Why Algorithm 3 applies to MDLM-type models

### What Algorithm 3 estimates (from your LLaDA tex)
- The paper uses a conditional likelihood estimator that averages masked-token conditionals over uniformly sampled mask sets:

```266:271:/Users/noesis/Code/plmentropyft/main.tex
\begin{align}
\label{eq:ppl-eval}
    -\mathbb{E}_{l, r_0, r_l} \left[\frac{L}{l} \sum_{i=1}^L \textbf{1}[r_l^i = \textrm{M}] \log p_{\theta}(r_0^i|p_0, r_l) \right],
\end{align}
```

- Algorithm 3 implements a Monte Carlo estimate of the same quantity by uniformly sampling an integer mask size `l` and a uniform `l`-subset of positions, then summing log‑probabilities only over those masked indices:

```818:832:/Users/noesis/Code/plmentropyft/main.tex
\begin{algorithm}[t!]
    \caption{Conditional Log-likelihood Evaluation of LLaDA}
    \label{alg:likelihood}
    \begin{algorithmic}[1]
        \REQUIRE mask predictor $p_{\theta}$, prompt $p_0$, response $r_0$, the number of Monte Carlo estimations $n_{mc}$
        \STATE $\text{log}\_\text{likelihood}=0 $
            \FOR{$i \gets 1$ to $n_{mc}$}
                \STATE $l \sim \{1, 2, \dots, L\}$ \hfill \# $L$ is the sequence length of $r_0$
                \STATE Obtain \( r_l \) by uniformly sampling \( l \) tokens from \( r_0 \) without replacement for masking
                \STATE $\text{log}\_\text{likelihood} = \text{log}\_\text{likelihood} + \frac{L}{l} \sum_{i=1}^L \textbf{1}[r_l^i = \textrm{M}] \log p_{\theta}(r_0^i|p_0, r_l) $
            \ENDFOR
        \STATE $\text{log}\_\text{likelihood} = \text{log}\_\text{likelihood} / n_{mc} $
        \STATE \textbf{Return} $\text{log}\_\text{likelihood}$
    \end{algorithmic}
\end{algorithm}
```

This is a pathwise MC estimator of Eq. (ppl‑eval): sample a uniform masked set, evaluate the product/sum of masked‑token conditionals, average.

### MDLM structure (from Sahoo et al.) matches Algorithm 3’s requirements
Algorithm 3 is valid when the reverse process:
1) is monotone reveal‑only (already revealed tokens never change),
2) provides per‑token conditionals given the current partially revealed sequence, and
3) uses a known (here: uniform) selection of which positions are masked/unmasked at evaluation time.

Your MDLM in `mainSahoo.tex` satisfies these via masking (absorbing prior), SUBS parameterization, and sequence‑wise factorization:

- Absorbing mask forward process (masking):

```295:301:/Users/noesis/Code/plmentropyft/mainSahoo.tex
In masked (i.e., absorbing state) diffusion, we set $\prior = \m$. At each noising step, $t$, the input $\x$ transitions to a `masked' state $\m$ with some probability.
If an input transitions to $\m$ at any time $t'$, it will remain in this state for all $t > t': q(\z_t \mid \z_{t'} = \m) = \cat(\z_t; \m).
The marginal of the forward process (\ref{eqn:interpolating_forward}) is given by 
q(\z_t| \x) = \cat(\z_t; \alpha_t \x + (1 - \alpha_t) \m).
```

- Monotone reverse (unmasked stays fixed) and no prediction of [MASK] (SUBS):

```343:352:/Users/noesis/Code/plmentropyft/mainSahoo.tex
Furthermore, we induce 2 key properties of the absorbing state diffusion process into our denoising model, $\x_\theta$: an unmasked token remains unchanged during reverse diffusion, and the clean input is never masked.
... we design the denoising network such that $\x_\theta[m] = 0$ (Zero Masking Probabilities) ...
Carry-Over Unmasking ... copy unmasked inputs.
```

- Sequence‑level factorization into per‑token conditionals given the current partially masked sequence:

```403:408:/Users/noesis/Code/plmentropyft/mainSahoo.tex
... conditioned on a sequence of latents $\z_t^{1:L},$ the denoising process factorizes independently across tokens, i.e., $p_\theta(\z_s^{1:L} \mid \z_t^{1:L}) = \prod_{\ell=1}^L p_\theta(\z_s^{\ell} \mid \z_t^{1:L}).$
Thus, we use a single model to compute $\denoise^\ell(\z_t^{1:L}, t)$ for each $\ell$ ... optimizing: $\sum_{\ell} \log \langle \denoise^\ell(\z_t^{1:L}, t), \x^\ell \rangle$.
```

- Only masked positions contribute at a given state (matches Algorithm 3’s per‑draw masked‑token sum):

```413:414:/Users/noesis/Code/plmentropyft/mainSahoo.tex
Although ~\Eqn{eqn:dif_loss_cont_subs_multidim} imposes a loss on all tokens, unmasked tokens don't contribute to the loss, as they are copied over by the denoising network due to “carry-over unmasking”.
```

- Practical evaluation/implementation explicitly examines only masked indices:

```421:424:/Users/noesis/Code/plmentropyft/mainSahoo.tex
... we evaluate $\mathrm{KL}[q(\z_s \mid \z_t, \x)\|p_\theta(\z_s\mid\z_t)]$ by examining only the masked token indices rather than comparing the full true and approximate posterior distributions.
```

Together, these guarantee the exact ingredients Algorithm 3 needs: a reveal‑only process, valid per‑token conditionals given a masked context, and additivity of log‑likelihood contributions over the currently masked set at each Monte Carlo draw.

And in LLaDA’s masked diffusion, the mask predictor is explicitly trained to predict masked tokens from the partially masked input:

```199:205:/Users/noesis/Code/plmentropyft/main.tex
The core of LLaDA is a \emph{mask predictor}, a parametric model \( p_\theta(\cdot|x_t) \) that takes \( x_t \) as input and predicts all masked tokens ...
\begin{align}
\label{eq:objective}
   \mathcal{L}(\theta)  \triangleq   -  \mathbb{E}_{t, x_0,  x_t} \left[\frac{1}{t} \sum_{ i = 1 }^L \textbf{1}[x_t^i = \textrm{M}] \log p_{\theta}(x_0^i|x_t) \right] , 
\end{align}
```

Moreover, the time‑free parameterization shows these conditionals depend only on the currently unmasked context:

```903:907:/Users/noesis/Code/plmentropyft/main.tex
q_{0|t}(x_s^i|x_t) = p_{\textrm{data}}(x_0^{i}|x_t^{\textrm{UM}}), \quad \forall i \textrm{ such that } x_t^{i} = \textrm{M},
where \( x_t^{\textrm{UM}} \) denotes the collection of unmasked tokens in \( x_t \) ...
```

- Known uniform selection in Algorithm 3: the algorithm uniformly samples the mask size `l` and a uniform `l`‑subset of positions, then sums log‑probabilities only over those masked positions at that draw. This matches MDLM’s semantics because the model exposes $p_\theta(\cdot\mid$ current masked state$)$ for any mask set; SUBS and factorization ensure revealed tokens stay fixed while masked tokens have proper conditionals.

```825:827:/Users/noesis/Code/plmentropyft/main.tex
\STATE $l \sim \{1, 2, \dots, L\}$ \hfill \# $L$ is the sequence length of $r_0$
\STATE Obtain \( r_l \) by uniformly sampling \( l \) tokens from \( r_0 \) without replacement for masking
\STATE $... + \frac{L}{l} \sum_{i=1}^L \textbf{1}[r_l^i = \textrm{M}] \log p_{\theta}(r_0^i|p_0, r_l)$
```

### Why only masked tokens contribute each MC draw
At each sample, the sum is gated by \(\mathbf{1}[r_l^i = \textrm{M}]\). In MDLM, carry‑over unmasking makes unmasked positions deterministic copies (no loss term), and implementation evaluates KL only on masked indices:

```413:414:/Users/noesis/Code/plmentropyft/mainSahoo.tex
... unmasked tokens don't contribute to the loss, as they are copied over by the denoising network ...
```

```421:424:/Users/noesis/Code/plmentropyft/mainSahoo.tex
... evaluate ... by examining only the masked token indices ...
```

### Conclusion
MDLMs (absorbing mask, monotone reveal‑only, per‑token conditionals given current partial context) satisfy all structural requirements behind Algorithm 3’s estimator. Because Algorithm 3 samples uniform mask‑sets and evaluates only masked‑token conditionals, it provides a valid MC estimate of the conditional log‑likelihood for MDLM‑type models as defined in Schiff et al. and instantiated by LLaDA.

### Why Algorithm 3 does not apply to the other model types in “Steering…”

- D3PM (general discrete diffusion with transition matrices):
  - Non‑monotone, parallel token updates: tokens can change among the entire vocabulary according to Q‑matrices; already set tokens are not guaranteed to remain fixed.

```227:231:/Users/noesis/Code/plmentropyft/mainSahoo.tex
D3PM ... introduces a framework with a Markov forward process $q(\z_t | \z_{t-1}) = \cat( \z_t ; Q_t \z_{t-1})$ ... inducing marginals $q(\z_t | \x) = \cat(\z_t ; \bar Q_t \x)$.
```

  - Consequence: Algorithm 3’s assumptions break (no monotone reveal‑only path; no uniform reveal order over masked positions), so you cannot rewrite $p_\theta(x)$ as the MDLM‑style expectation over uniform reveal orders or evaluate only “newly revealed” masked‑token factors. D3PM likelihood is instead handled via its variational bound (ELBO).

- UDLM (uniform‑noise discrete diffusion):
  - Non‑monotone denoising: tokens can be edited multiple times during the reverse process; the process is intentionally revisable for guidance/control.

```259:266:/Users/noesis/Code/plmentropyft/mainMDLM.tex
\textbf{Uniform Noise Forward Process} ... When letting $\prior = \uniform$, the input $\x$ transitions to a random state with some probability at each time step. 
```

```261:262:/Users/noesis/Code/plmentropyft/mainMDLM.tex
Crucially, after $\x$ changes once, it can do so again.
```

  - Consequence: the monotone reveal‑only requirement is violated; a per‑step factor is not “reveal a masked token’s true value and keep it forever,” so the page‑19 estimator (uniformly sample a mask set and sum only masked‑token conditionals) no longer corresponds to the model’s joint. UDLM therefore uses its own ELBO (and continuous‑time refinement) rather than Algorithm 3’s masked‑token MC estimator.

In short: only the absorbing‑mask, monotone MDLM fits Algorithm 3’s structure. D3PM and UDLM allow tokens to change non‑monotonically (and, for D3PM, in parallel with general Q), so their exact likelihoods are not captured by Algorithm 3’s uniform‑mask‑set MC estimator; they instead rely on diffusion ELBOs or alternative estimators.


