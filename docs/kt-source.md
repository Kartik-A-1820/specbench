# specbench — Knowledge Transfer

Source text for the handover document. `scripts/build_kt_docx.py` reads this
file plus the measured results and produces the .docx. Numbers are never typed
into this file: placeholders of the form `{{RESULTS:...}}` are substituted from
`results/summary.json` at build time.

---

## PART I — WHY THIS IS WORTH DOING

### 1.1 Autoregressive decoding is memory-bandwidth-bound, not compute-bound

Generating one token from a transformer at batch size 1 requires reading every
parameter of the model out of GPU memory exactly once. It requires almost no
arithmetic by comparison. That asymmetry is the whole reason speculative
decoding exists.

Take the numbers for this project's target model on this project's GPU.

TinyLlama-1.1B in FP16 is 2.20 GB of weights. The GTX 1650 Ti Mobile has a
memory bandwidth of about 192 GB/s. Reading the weights once therefore takes at
minimum

    2.20 GB / 192 GB/s = 11.5 ms

which caps single-stream decoding at roughly 87 tokens per second before any
arithmetic, kernel launch overhead, attention over the KV cache, or Python is
accounted for. The measured baseline is well below that ceiling, as it always
is, but the ceiling is set by bandwidth.

Now count the arithmetic. A forward pass over a single token costs about 2N
FLOPs for N parameters — one multiply and one add per weight — so about
2.2 GFLOP. The 1650 Ti does roughly 8 TFLOP/s in FP16. The arithmetic takes
about 0.3 ms. The memory traffic takes 11.5 ms.

**The GPU spends roughly 97% of a decode step waiting for weights to arrive and
about 3% of it computing.** The arithmetic units are idle almost all of the
time.

The standard way to express this is arithmetic intensity: FLOPs performed per
byte read. At batch size 1 the transformer's matrix multiplications are
matrix-vector products, and a matrix-vector product does 2 FLOPs per element
read — an arithmetic intensity of 1 FLOP/byte in FP16. The 1650 Ti's ridge
point, where it stops being bandwidth-limited and starts being compute-limited,
is around 8000 GFLOP/s ÷ 192 GB/s ≈ 42 FLOP/byte. Batch-1 decoding sits a
factor of forty on the wrong side of that line.

### 1.2 Therefore: verifying k tokens costs about the same as generating one

If a forward pass is dominated by the cost of streaming the weights, then a
forward pass over k tokens at once costs about the same as a forward pass over
one token — because the weights are read once either way. The matrix-vector
products become matrix-matrix products with k columns. The arithmetic goes up
by a factor of k; the memory traffic does not.

That holds until k grows large enough to push arithmetic intensity past the
ridge point, at which point the pass really does start costing k times as much.
For this hardware that crossover is far above the speculation depths that are
useful, so within the range K = 1..8 a verification pass over K+1 tokens costs
very nearly what a pass over one token costs.

This is the exploitable asymmetry:

- **Generating** k tokens autoregressively costs k forward passes, because each
  token must be sampled before the next can be conditioned on it.
- **Checking** k already-written tokens costs one forward pass, because they can
  all be fed at once and the causal mask makes position j's output depend only
  on positions ≤ j.

Speculative decoding buys the second with a cheap model and pays for it in
correctness-preserving rejections.

### 1.3 Where the speedup comes from, precisely

Let α be the mean number of tokens emitted per round (the "acceptance length").
Let c be the cost of one draft forward pass relative to one target forward pass.
A round costs one target pass plus K draft passes, so

    time per token  =  (1 + K·c) / α   target-pass-equivalents

and the speedup against the baseline's one target pass per token is

    speedup  =  α / (1 + K·c)

Three things fall straight out of that expression, and all three are visible in
the measured results:

1. **α is the numerator, so acceptance rate is the thing to maximise.** This is
   what the EAGLE head is for.
2. **K appears in the denominator multiplied by c.** Increasing K raises α, but
   with diminishing returns (each extra proposal is conditioned on all previous
   ones being accepted, so its acceptance probability is the product of
   increasingly many terms) while the cost K·c grows linearly. There is an
   optimum, and past it speedup falls.
3. **c matters as much as α.** A draft model that is half as accurate but ten
   times cheaper can win. This is the argument for a small draft head over a
   small draft model.

---

## PART II — THE ALGORITHM AND ITS CORRECTNESS

### 2.1 One round

Write p for the target model's next-token distribution and q for the draft's,
both after identical temperature / top-k / top-p processing.

1. The draft proposes x₁…x_K autoregressively, sampling xᵢ ~ qᵢ, where qᵢ is
   conditioned on the prefix plus x₁…x_{i−1}.
2. The target runs **one** forward pass over the proposals and returns
   p₁…p_{K+1}.
3. For i = 1…K in order: draw r ~ Uniform(0,1) and accept xᵢ if

       r < min(1, pᵢ(xᵢ) / qᵢ(xᵢ))

   On the first rejection, emit a token drawn from the residual distribution

       p'ᵢ(x) = normalize( max(0, pᵢ(x) − qᵢ(x)) )

   and end the round.
4. If all K were accepted, emit one extra token drawn from p_{K+1}. This is the
   bonus token, and it is free: p_{K+1} came out of the same verification pass.

A round therefore emits between 1 and K+1 tokens and always emits at least one.
That lower bound is what makes the loop terminate.

### 2.2 Deriving the residual

The acceptance rule and the residual are not two independent design choices.
Fix the residual and the acceptance rule is determined, and vice versa. Here is
the derivation, which is the thing to be able to reproduce on a whiteboard.

Consider one position. The draft proposes x ~ q. We accept with probability
min(1, p(x)/q(x)). The probability that the round emits token x *via
acceptance* is

    P(propose x and accept it) = q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))

Summing over x, the total probability of accepting anything is

    P(accept) = Σₓ min(p(x), q(x))

which is exactly 1 − TVD(p, q), the overlap between the two distributions. So
the acceptance rate is a direct measure of draft–target agreement, and it is a
quantity with an interpretation, not a tuning knob.

We want the overall emitted token to be distributed as p. Acceptance already
delivers min(p(x), q(x)) of the mass at each x. The shortfall at x is

    p(x) − min(p(x), q(x)) = max(0, p(x) − q(x))

so on rejection — which happens with probability 1 − Σₓ min(p,q) = Σₓ max(0,
p−q) — we must emit x with probability proportional to max(0, p(x) − q(x)).
That is the residual, and the normalising constant is exactly the rejection
probability. Nothing was chosen; it was forced.

### 2.3 The proof that the output distribution is preserved

For any token x, the probability that a given position emits x is

    P(emit x) = P(accept, and the accepted token is x)
              + P(reject) · P(residual draws x)

    P(accept and emit x) = min(p(x), q(x))

    P(reject)            = Σ_y max(0, p(y) − q(y))

    P(residual draws x)  = max(0, p(x) − q(x)) / Σ_y max(0, p(y) − q(y))

The rejection probability cancels against the residual's normaliser:

    P(emit x) = min(p(x), q(x)) + max(0, p(x) − q(x))

Both branches of the min/max split give the same answer:

- if q(x) ≥ p(x):  min = p(x), max = 0        → p(x)
- if q(x) <  p(x):  min = q(x), max = p(x)−q(x) → p(x)

So P(emit x) = p(x) exactly, **for any q whatsoever**. The draft distribution
does not appear in the result.

Two consequences to state explicitly in an interview, because they are the
points that get probed:

- The guarantee is per-position and unconditional in q. A badly trained,
  adversarial, or even deliberately hostile draft model changes the acceptance
  *rate* and therefore the speed, and cannot change the output distribution.
- The bonus token is drawn from p_{K+1} directly, not through the accept/reject
  rule, so it is trivially correctly distributed. The subtlety is that it is
  only emitted conditional on all K proposals having been accepted, and
  p_{K+1} is conditioned on exactly that same prefix — which is the prefix that
  actually happened. The conditioning is consistent.

### 2.4 Expected acceptance length

Let β be the per-position acceptance probability, assumed roughly constant
across positions. Position i is only reached if positions 1…i−1 were all
accepted, so the number of accepted proposals is geometric-with-a-cap and

    E[tokens per round] = Σ_{i=0}^{K} βⁱ = (1 − β^{K+1}) / (1 − β)

The +1 in the exponent is the bonus token. This is where the diminishing return
in K comes from: the marginal contribution of the K-th proposal is β^K, which
decays geometrically, while its cost c is constant. Setting the derivative of
α/(1 + K·c) to zero gives the optimal K, and it depends on both β (a property
of the model pair and the text domain) and c (a property of the hardware). It
cannot be assumed; it has to be measured. That is why the benchmark sweeps K
rather than picking a value from a paper.

---

## PART III — IMPLEMENTATION DECISIONS

### 3.1 One target forward pass per round: the pending-token block

This is the single most consequential implementation detail in the project, and
the one most likely to be got wrong.

The accept/reject rule needs K+1 target distributions per round: p₁…p_K to
judge the proposals and p_{K+1} for the bonus token. But a forward pass over
the K proposals returns only K logit rows, and the missing one is p₁ — the
distribution over the *first* proposal, conditioned on the prefix alone. It is
determined before any proposal has been fed, so it is not in the block.

The obvious repair is a second, single-token target forward pass each round to
obtain it. That repair costs a factor of two on the dominant term. Revisiting
the speedup expression with two target passes per round:

    speedup = α / (2 + K·c)

Substituting this project's measured values at K = 4 — α = 1.767 and c = 0.069
(both derived in Part VII) — gives

    1.767 / (2 + 4 × 0.069) = 0.78

a **slowdown**. An implementation with this shape passes every correctness test
in the suite, produces exactly the right output distribution, and is slower than
the baseline it exists to beat. It is the standard way to build a speculative
decoder that does not speed anything up, and because nothing fails, the only way
to find out is to measure.

The fix is to carry a **pending token**: the token emitted at the end of the
previous round, committed to the output but deliberately withheld from both KV
caches. It heads the next round's block, so the target is fed

    [pending, x₁, x₂, …, x_K]      (K+1 positions, one pass)

and because output row j is the distribution over whatever follows block
position j:

    row 0 → p₁      (over x₁, given prefix + pending)
    row i → p_{i+1}
    row K → p_{K+1} (the bonus distribution)

All K+1 distributions, one pass. The extra token in the block costs essentially
nothing for the reason established in Part I: the pass is bandwidth-bound, so
its cost is set by the weights, not by the sequence length.

Cache bookkeeping follows from the same idea. Entering a round both caches hold
`committed_len` positions and exclude `pending`. After accepting a proposals the
committed prefix is `committed_len + 1 + a`, where the +1 is `pending` itself —
always correct, because it was emitted rather than proposed. The draft needs
only `pending` plus x₁…x_{K−1} to produce all K proposals, so it runs one
position short and is topped up with a single extra draft call in the one case
where that matters, a = K.

### 3.2 KV cache rollback

After a rejection at position i, the target's KV cache contains entries for
proposals that were never emitted. They must be removed, or every subsequent
token is conditioned on text the model never produced.

Rollback truncates the sequence dimension of every layer's K and V tensors to
the accepted prefix length. It is a slice, so it is cheap.

The reason this deserves a dedicated test is that **getting it wrong does not
crash anything.** Attention still runs, logits still look like logits, and the
model still emits fluent, plausible text — conditioned on a phantom prefix.
There is no exception, no NaN, no obviously wrong output. The only way to catch
it is a test constructed so that a stale cache must produce a different answer.

The test in `tests/test_kv_rollback.py` therefore does three things a weaker
version would not:

1. It pollutes the cache with a branch of *deliberately unrelated* tokens.
   Polluting with the same tokens that will later be regenerated makes a broken
   rollback and a correct one indistinguishable.
2. It asserts that the polluted context actually changes the model's next-token
   prediction, so the test is known to have teeth on this model and prompt
   rather than being vacuously satisfiable.
3. Separately, it drives the real speculative loop, asserts that rejections
   genuinely occurred at more than one offset, and only then compares against
   the baseline.

### 3.3 Identical logits processing on both sides

Temperature, top-k and top-p must be applied identically to p and q before the
ratio p(x)/q(x) and the residual are computed. This is not a stylistic
preference; the proof in 2.3 is a statement about the two distributions that
are actually compared. If q is warped and p is not, the algorithm is provably
sampling from something, but not from the warped p the user asked for.

The enforcement mechanism is that there is exactly one function,
`logits_to_probs`, and both paths call it. There is no second code path that
could drift.

### 3.4 Reproducibility

Every random draw — draft sampling, the acceptance coin, the residual sample,
the bonus token — goes through one seeded generator.

The subtle failure this replaced: `torch.multinomial` silently ignores a CPU
generator when the probability tensor is on CUDA. The original code guarded
with `if generator is not None and tensor.device.type == "cpu"`, so on GPU the
generator was skipped and the global RNG was used. Runs did not reproduce, but
`seed` was still written into every result record, which is worse than not
recording it: it asserts a property the artefact does not have. The generator
is now built on the sampling device, and a device mismatch raises rather than
falling back.

---

## PART IV — THE EAGLE DRAFT HEAD

### 4.1 What it is

Instead of a separate small language model, EAGLE trains a small head that
autoregresses on the **target model's own last-layer features**. Writing f_t for
the target's feature at position t:

    head( f_{t−1}, embed(token_t) )  →  f̂_t  →  target's frozen LM head  →  q

The head is one or two Llama decoder layers plus an input projection, and it
reuses the target's frozen token embedding and LM head rather than owning its
own copies.

### 4.2 Why feature-level drafting beats a separate draft model

Four arguments, in roughly descending order of how much they matter:

**It drafts in the target's own representation space.** A separate draft model
is a different network that happened to be trained on similar data; its
disagreements with the target are arbitrary. The head is trained specifically to
predict what the target's next feature will be, so its errors are correlated
with the target's own uncertainty rather than independent of it. Higher α for
the same parameter count.

**It shares the expensive parts.** The token embedding and LM head for a
32000-vocabulary, 2048-dimensional model are about 131 M parameters each — more
than the entire llama-68m draft model. The head borrows both, frozen. A separate
draft model has to carry its own, and pays for them in VRAM and in bandwidth on
every draft step.

**Features are lower-entropy than tokens.** Sampling a token throws away
everything the distribution knew except the one index that was drawn. Passing a
feature vector forward preserves the target's uncertainty into the next draft
step, so error accumulates more slowly over K steps. This is EAGLE's central
claim and it is why acceptance holds up at larger K.

**c is smaller.** One or two decoder layers against a 22-layer target is a much
better cost ratio than a 2-layer separate model that also has to run its own
embedding and vocabulary projection. Since speedup is α/(1 + K·c), lowering c
helps exactly as much as raising α.

The cost is that a head is target-specific and must be trained, whereas a draft
model can be downloaded. That is a real trade-off, and on this project it is the
one that decided the outcome.

**Those four arguments are the case for EAGLE. This project measured the
opposite.** Every trained head came in below 1.0x — slower than running the
target alone — and lost to the downloadable llama-68m on both terms of
alpha / (1 + K*c) simultaneously. Part VII gives the numbers and the diagnosis.

The short version, stated here so this section is not read as a claim the
results contradict: arguments two and four (shared embeddings, lower c) are
about cost, and both turned out to be wrong on this hardware — the head is
roughly twice as expensive per draft step as the small draft model, because it
projects from hidden 2048 through a 32,000-row vocabulary in FP32 while
llama-68m projects from hidden 768 in FP16. Arguments one and three are about
quality, and they are conditional on the head actually being trained; on a
250,000-token budget it memorised its corpus instead. None of this shows the
method is wrong. It shows that the method's advantages are contingent, and that
this project could not afford the thing they are contingent on.

### 4.3 The pairing that has to match

The head is trained on `(f_{t−1}, embed(token_t)) → f_t`. Inference must use the
same pairing. Feeding `(f_t, embed(token_t))` instead — one position later than
anything seen in training — is a bug this project actually had.

What makes it dangerous is the failure mode. The accept/reject rule still
guarantees correct output, so nothing looks broken: no crash, no garbage text,
no failing correctness test. The only symptom is that acceptance collapses and
EAGLE measures no faster than the baseline, which is easy to misread as "EAGLE
does not help on this hardware" rather than "the head is being fed the wrong
input."

This generalises to a point worth making in an interview: in speculative
decoding, **every draft-side bug is a performance bug, not a correctness bug**,
because the sampler absorbs it. That is a genuine safety property, and it also
means the draft side needs its own tests, since the correctness suite will
never catch anything there.

### 4.3.1 A worked example of exactly that failure mode

This project hit it, and the way it was caught is worth reproducing.

The first head trained to a final cross-entropy of **0.053**. That looks like an
excellent result. It is impossible.

The head is distilling the target model's own next-token distribution, so its
loss has a known reference point: the target's own cross-entropy on the same
data. Decoding the cached features through TinyLlama's frozen LM head and
scoring them against the stored next tokens gives that directly. Over 250
random cached documents (49,904 token positions) it is **2.56** (perplexity
13.0).

A head that has been fitted to this exact data for several epochs can
legitimately edge a little below the teacher's zero-shot figure — that is
ordinary overfitting on the training set, and the correctly-trained head in
this project lands at 2.60, just above it. What is not explicable is 0.053:
roughly fifty times below the floor. No amount of fitting produces that. It
means the head is not predicting at all, it is copying.

The cause was one missing argument. `EagleDraftHead.forward` passed
`attention_mask=None` into `LlamaDecoderLayer`. A bare decoder layer built from
a plain `LlamaConfig` uses the *eager* attention path, and eager attention
applies no mask at all when the mask is None — so attention was bidirectional.
The training labels make that fatal: the label for position t is the input token
at position t+1, so the head learned to read the answer off the future rather
than predict it.

What makes this the archetype of a draft-side bug:

- Nothing crashes. No exception, no NaN, no failing correctness test.
- Training looks *better* than expected, not worse, so the natural reaction is
  satisfaction rather than suspicion.
- At inference the head runs one position at a time with no future to read, so
  the learned strategy is worthless and acceptance collapses.
- The output remains exactly correct throughout, because the accept/reject rule
  absorbs a bad q. The only visible symptom is "EAGLE gives no speedup", which
  reads as a legitimate negative finding about the method or the hardware.

That last point is the real hazard: the bug's signature is a plausible,
well-supported, wrong conclusion.

Three defences were added, in increasing order of how much they generalise:

1. An **explicit causal mask** constructed inside `forward()`, so behaviour no
   longer depends on which attention implementation the config selects.
2. A **structural test** (`tests/test_eagle_head.py`): perturb the input at
   position j, require every output at t < j to be bit-identical. It was
   verified by reverting the fix — three tests fail without the mask and pass
   with it. It runs on CPU in seconds.
3. A **plausibility floor on the loss**. Training warns if the final
   cross-entropy falls below half the measured teacher floor, and the benchmark
   sweep refuses to run against any head so flagged, or against any head whose
   recorded architecture version does not match the current one.

The general lesson is the one worth stating out loud: **a distillation loss has
a known lower bound, and checking against it is nearly free.** Any student model
that beats its teacher at imitating the teacher is leaking, and that check
generalises to every distillation setup, not just this one.

### 4.4 Offline hidden-state caching: a hardware-forced design

Training the head with the target resident in VRAM does not fit in 3.4 GB:

| Component | Size |
|---|---|
| Target weights (frozen, FP16) | 2.20 GB |
| Head parameters (~60 M, FP32) | 0.24 GB |
| Gradients | 0.24 GB |
| AdamW first and second moments | 0.48 GB |
| Activations | ~0.30 GB |
| **Total** | **~3.5 GB — over budget** |

Note that the optimiser state alone is three times the size of the head. AdamW
keeps two FP32 moments per parameter, so training a model costs roughly 4× its
parameter memory before activations.

The workflow is therefore split: run the target over the corpus once, write
last-layer features to disk in FP16 at 4 KB per token, free the target, then
train against the cached tensors. Without the target body:

| Component | Size |
|---|---|
| Head + gradients + AdamW state | ~0.96 GB |
| Frozen embedding + LM head (FP16) | ~0.13 GB |
| Activations | ~0.20 GB |
| **Total** | **~1.29 GB** |

This is the kind of decision worth presenting as a decision rather than as a
detail: it is not an optimisation, it is the only shape of the workflow that
fits, and it costs disk and a second pass over the corpus to buy the VRAM.

---

## PART V — HARDWARE CONSTRAINTS AND WHAT THEY FORCED

### 5.1 The device

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 Ti Mobile (TU117) |
| Compute capability | SM 7.5 (Turing) |
| Physical VRAM | 4.29 GB total, ~3.4 GB usable after WDDM overhead |
| Memory bandwidth | ~192 GB/s |
| Driver | 610.62 |

### 5.2 Model pairs that did not fit

Both models must share a vocabulary — p and q have to be distributions over the
same index set for the ratio to mean anything — which restricts the search to
same-family pairs.

**Qwen2.5-1.5B-Instruct + Qwen2.5-0.5B-Instruct.** The natural modern pair with
strong acceptance. Target weights 1.54 B × 2 B = 3.09 GB; draft 0.494 B × 2 B =
0.99 GB; 4.08 GB of weights alone against a 3.4 GB budget. Rejected before the
KV cache is even considered.

**Pythia-1.4B + Pythia-160M.** The classic pair from the speculative decoding
literature. Weights 2.83 + 0.32 = 3.15 GB, which leaves 0.25 GB. Worse, Pythia
uses full multi-head attention with head_dim 128:

    2 × 24 layers × 16 heads × 128 × 2 bytes = 196,608 B ≈ 192 KB per token

At 1024 context that is 0.197 GB of KV cache for the target alone, taking the
total to ~3.35 GB. It fits only at a very short context and leaves nothing for
Phase 2 training state. Rejected.

**TinyLlama-1.1B-Chat + llama-68m — selected.** 2.20 + 0.14 = 2.34 GB of
weights. The decisive property is TinyLlama's grouped-query attention: 4 KV
heads against 32 query heads, so

    2 × 22 layers × 4 KV heads × 64 × 2 bytes = 22,528 B ≈ 22 KB per token

which is roughly 46 MB at 2048 context — about an eighth of what a full-MHA
model of the same width would need. **GQA is the reason this configuration
fits at all**, and it is worth being able to say so with the arithmetic.

### 5.3 A measurement subtlety worth knowing

The VRAM probe originally reported the target KV cache as ~3.7× its predicted
size. The prediction was right; the measurement was contaminated. The probe
measured allocated memory before and after a prefill and attributed the whole
difference to the KV cache — but it was still holding a reference to the
returned logits tensor, which at 2048 positions × 32000 vocabulary in FP16 is
about **131 MB, nearly three times the KV cache itself**.

Two lessons. First, at long context and large vocabulary the prefill logits
tensor is the largest transient in the pipeline, not the KV cache. Second, a
memory measurement that does not account for what the measuring code itself is
still holding will blame the wrong component.

### 5.4 Other constraints from SM 7.5

**No FlashAttention-2** — it requires SM 8.0. PyTorch SDPA is used instead;
it is available on Turing and handles FP16 correctly.

**FP16, not bfloat16.** Turing has no bf16 hardware. Worth noting that
`torch.cuda.is_bf16_supported()` returns True on Turing because it reports
emulation, so it is not a usable test for whether bf16 is a good idea — an
assertion built on it fails spuriously.

**FP16 has ~11 bits of mantissa**, which is why the acceptance ratio
p(x)/q(x) and the residual are computed in FP32. In the tail of the
distribution, where both p and q are small, FP16 rounding is a large relative
error on exactly the quantity the accept/reject decision depends on.

---

## PART VI — BENCHMARK METHODOLOGY

### 6.1 Interleaved multi-arm with rotation

Every condition is an arm: the baseline, and one per speculation depth. For each
prompt, every arm runs once, back to back, and the arm order is rotated by one
position from prompt to prompt.

**Why not blocked (all baselines, then all speculative)?** A GTX 1650 Ti in a
laptop chassis throttles from ~1485 MHz to roughly 800–1000 MHz within the first
five to ten minutes of sustained load. A blocked design charges that entire
drift to whichever condition ran second. The bias is large, systematic, and
points in whichever direction the running order happened to choose.

**Why rotate as well as interleave?** Interleaving alone leaves order within a
prompt confounded with arm identity: whichever arm always runs first pays a
colder cache, whichever runs last sits at the local thermal peak. Rotation
spreads that evenly, and `arm_position` is recorded so the residual effect can
be estimated afterwards rather than assumed away.

**Why one baseline shared across arms?** Running a separate baseline per K would
be six or seven measurements of the same quantity taken at six or seven
different points in a thermal ramp — slower and noisier than measuring it once
per prompt. Sharing it is also what makes the paired bootstrap valid.

### 6.2 Bootstrap confidence intervals, and why not point estimates

A point estimate of speedup on 29 prompts, from a thermally noisy laptop GPU, is
a number with unknown precision. Reported without an interval, it invites the
reader to treat noise as signal — and if two configurations differ by 3%, a point
estimate cannot say whether that difference is real.

Two specific choices:

**Ratio of means, not mean of ratios.** speedup = mean(spec_tps) /
mean(baseline_tps). The mean-of-ratios estimator is biased upward by Jensen's
inequality applied to the reciprocal, and its sampling distribution is harder to
characterise. Ratio of means is standard in systems benchmarking.

**Paired resampling.** Each bootstrap replicate draws the *same* prompt indices
for both arms. Prompt difficulty affects both conditions in the same direction,
so pairing removes that shared variance and gives a tighter, honest interval.
Resampling the two arms independently would inflate the interval with variance
that the experimental design already controlled for.

The bootstrap is used rather than a parametric interval because the sampling
distribution of a ratio of means is not normal, and 29 prompts is not enough for
asymptotics to rescue it.

### 6.3 Latency: two series, no synthesised percentiles

Speculative decoding has bursty latency. A verification pass emits several
tokens at the same instant; then the system pauses for the next round. The
distribution is bimodal — a spike at 0 ms and a second mode at the round
duration — and a mean hides exactly the property a user perceives as stutter.

Two series are recorded separately:

- **`round_latencies_ms`** — wall-clock duration of one round. For the baseline a
  round is one token, so this is conventional inter-token latency.
- **`token_arrival_gaps_ms`** — interval between consecutive tokens becoming
  available. Zero for every token after the first in a speculative round,
  because they genuinely arrived together.

p50 of the arrival-gap series can legitimately be 0 ms while p90 is large. That
bimodality is the finding, not a defect.

Timestamps are taken inside the generators, immediately after a `.item()` call
that has already forced a device synchronisation. No extra `cuda.synchronize()`
is inserted in the measured region. This matters for fairness: the baseline
synchronises once per token and the speculative path once per round, so adding a
barrier per timestamp would tax the baseline more heavily and manufacture a
speedup.

**What this replaced.** The original harness recorded `wall_clock / n_tokens`
repeated n_tokens times. Every percentile of that series equals the mean by
construction — p50 = p90 = p99, standard deviation exactly zero — and the
burstiness that is the entire reason to report percentiles for this algorithm is
invisible in it. Percentiles computed from a mean are not measurements.

### 6.4 What the correctness suite actually proves

| Test | What it establishes |
|---|---|
| `test_sampler_synthetic.py` | The accept/reject rule reproduces p on synthetic distributions, including adversarial near-disjoint pairs, at 10⁶ Monte Carlo trials per case, judged by chi-square and total variation distance. No models involved, so a failure is unambiguously the sampler. |
| `test_greedy_equivalence.py` | At temperature 0 the speculative path is token-identical to the baseline over 10 prompts × 128 tokens at K ∈ {1,2,4,8}. This is the strongest single check of indexing: any off-by-one in the logits map or the cache shows up immediately as a divergence. |
| `test_kv_rollback.py` | Rollback erases rejected branches, verified against a deliberately divergent branch and through the real loop under forced rejections. |
| `test_vram_budget.py` | Weight and cache footprints stay within the measured budget. |

The greedy-equivalence test deserves a caveat that is worth raising before an
interviewer does. Greedy FP16 equivalence is legitimately fragile: a batched
forward pass over K+1 positions and a sequence of single-token passes use
different cuBLAS kernels and different reduction orders, so their logits can
differ in the last bits. When two tokens are near-tied, that is enough to flip
an argmax. Passing is strong evidence of correct indexing; a rare failure would
warrant checking the logit gap at the divergence point before assuming a bug.

---

## PART VII — RESULTS

{{RESULTS:PHASE1}}

{{RESULTS:LATENCY}}

{{RESULTS:DOMAIN}}

{{RESULTS:PHASE2}}

{{RESULTS:EAGLEVSDRAFT}}

{{RESULTS:COSTMODEL}}

### Why 1.32x and not the 2-3x the literature reports

The published speculative decoding results are mostly 2x to 3x. This project
reached 1.32x. The gap is worth accounting for precisely, because two different
explanations are available and only one of them is what the data says.

The first candidate is the hardware: this machine is slow, so speculation
cannot help as much. That is the intuitive answer and it is wrong, or at least
irrelevant. Speedup is a ratio, and a slow target model makes both the numerator
and the denominator slow. What matters is not how fast the device is but how the
two terms of `alpha / (1 + K·c)` come out on it.

The second candidate is acceptance, and this is the one the measurements
support. The section below fixes both terms against the measured data and shows
that c is not the binding constraint here — that even a hypothetical drafter
costing nothing at all would not have reached 2x with this draft model. What
caps this configuration is how often llama-68m and TinyLlama agree.

That distinction matters for what to do next, which is the practical value of
having the decomposition rather than just the headline. It says the effort
belongs on the numerator.

{{RESULTS:ROOFLINE}}

---

## PART VIII — LIMITATIONS

**Single device, single session.** All numbers come from one GTX 1650 Ti Mobile.
Confidence intervals capture within-session variance across prompts; they do not
capture session-to-session variation in thermal state. A rerun on a cold machine
would produce a different absolute throughput.

**Thermal throttling is mitigated, not eliminated.** The interleaved design
prevents drift being attributed to one arm. It does not stop the GPU from being
slower at the end of a run than at the start. Temperature and clock are recorded
per run so the effect can be quantified after the fact.

**Small prompt set.** 31 prompts, of which 2 are discarded as warmup. Enough for
a usable bootstrap interval, not enough to characterise acceptance across genres
finely.

**128 generated tokens per run.** Long-context behaviour is not measured.
Acceptance is known to vary with position, and KV cache growth changes the
bandwidth picture at multi-thousand-token contexts.

**Batch size 1 only.** The entire premise — that decoding is bandwidth-bound —
weakens as batch size grows, because a batched decode amortises the weight read
across sequences. At large batch the arithmetic intensity rises past the ridge
point, verification stops being nearly free, and speculative decoding's benefit
shrinks. Everything here is a statement about single-stream latency.

**EAGLE head training is small, and this dominates the Phase 2 result.** A
250 K-token feature cache and three epochs is orders of magnitude below what the
EAGLE papers use. The heads overfit it: acceptance is about 0.28 on their own
training documents and about 0.05 on held-out benchmark prompts. Every Phase 2
number is therefore a measurement of that training budget, not of the method.
They are a floor on what the architecture can reach, not an estimate of it.

**EAGLE training corpus is narrow.** wikitext-2 plus mbpp: general prose and
short Python. The benchmark prompt set contains idioms underrepresented there,
which compounds the overfitting above.

**The head runs in FP32.** Training was done in FP32 and inference kept it, so
each draft step pays roughly double the arithmetic it needs, and the head's cost
ratio c lands at 0.11 to 0.16 against llama-68m's 0.064 to 0.088. This is an
implementation choice, not a property of the method, and it is not something the
measured numbers should be read as indicting.

**No cross-hardware validation.** The VRAM arithmetic and model selection were
validated on this machine only.

**One target model, so the cost decomposition rests on two points.** The
draft-cost analysis separates a per-byte term from a per-layer term using two
models, which is two observations for two parameters and therefore zero degrees
of freedom. The decomposition is reported as a bracket — c falls between what
pure bandwidth predicts and what pure per-layer cost predicts — rather than as a
fitted split, because two points cannot support a fitted split. Measuring a
third model at a different depth-to-size ratio would turn the bracket into an
estimate.

**One draft model in the sampling sweep.** Every conclusion about acceptance is
a conclusion about llama-68m drafting TinyLlama. That the two are not the same
model family is the leading explanation for the low alpha, but it is an
explanation, not a measurement: no same-family pair was benchmarked, so the
size of the effect is unquantified here.

---

## PART IX — ANTICIPATED INTERVIEW QUESTIONS

**Q: Why is the output distribution preserved? Walk me through it.**

Reproduce section 2.3. The key move is that the rejection probability in the
second term cancels exactly against the residual's normalising constant, leaving
min(p,q) + max(0, p−q), and that expression equals p(x) in both branches of the
comparison between p(x) and q(x). Emphasise that q does not appear in the
result: the guarantee holds for *any* draft distribution, which is why a bad
draft costs speed and never quality. If pressed on where the residual comes
from, derive it as the shortfall rather than presenting it as a definition.

**Q: What happens when acceptance is low?**

α → 1: every round emits one token, from the residual. The work done is one
target pass plus K draft passes to produce what the baseline produces in one
target pass. Speedup = 1/(1 + K·c) < 1, so it is a genuine slowdown proportional
to the wasted draft work. Output is still exactly correct. Two mitigations worth
naming: reduce K (the loss is bounded by K·c), or adapt K online from a running
estimate of the acceptance rate, which is what production systems do. The
important framing is that the downside is bounded and predictable, while the
upside is not paid for in quality.

**Q: How does acceptance interact with temperature?**

Acceptance probability is Σ min(p, q) = 1 − TVD(p, q), so anything that changes
the overlap between the warped distributions changes it.

At temperature → 0 both distributions collapse toward one-hot at their argmaxes.
Acceptance becomes an indicator: 1 if the argmaxes agree, 0 otherwise. It is
bimodal, and it depends entirely on how often the draft picks the same top token.

Raising temperature flattens both distributions, which generally *increases*
overlap, so acceptance tends to rise with temperature. This surprises people who
expect more randomness to mean more disagreement — but the rule compares
distributions, not samples, and flatter distributions overlap more.

Top-k and top-p complicate this. They truncate support, and if q's support is
truncated somewhere p's is not, q(x) can be zero where p(x) is not, forcing a
rejection. This is exactly why both distributions must be warped identically
before comparison. The honest answer is that the effect is real, direction is
usually "warmer is more accepting", and this project measures both temperature
0 and 1 rather than arguing about it.

**Q: Why did this configuration win?**

Point at speedup = α/(1 + K·c) and show the measured sweep. Speedup rises with K
while the acceptance gain outpaces the added draft cost, peaks, and then falls —
because the marginal proposal contributes β^K, decaying geometrically, while its
cost is constant. Then be specific about what makes the optimum land where it
does on *this* hardware: c is set by the draft/target cost ratio, and both
models are small enough that fixed per-launch overhead is a non-trivial part of
c, which pushes the optimum lower than on a machine where the target dominates
more completely.

**Q: What would change on an A100?**

Several things, and they do not all point the same way.

*More VRAM (40–80 GB)* removes the binding constraint entirely. The rejected
Qwen and Pythia pairs become available, as do far larger targets. Larger targets
generally favour speculative decoding, because the draft/target cost ratio c
falls.

*Much higher bandwidth (~2 TB/s vs 192 GB/s)* raises the absolute ceiling but
does not change the bandwidth-bound character of batch-1 decoding — the ridge
point moves too. The qualitative argument survives.

*Fixed overheads matter more.* At A100 speeds a forward pass over a 1.1 B model
takes a fraction of a millisecond, and kernel launch latency and Python overhead
become a larger share of c. With CUDA graphs those can be amortised; without
them, the measured optimum K could move.

*bf16 and FlashAttention-2 become available* (SM 8.0), so the FP16-only and
SDPA-only constraints disappear.

*Batching changes the premise.* An A100 would realistically serve batched
requests, and as batch grows the weight read amortises across sequences,
arithmetic intensity crosses the ridge point, and verification stops being
nearly free. Speculative decoding's benefit narrows substantially at high batch.
The correct framing is that this is a **latency** optimisation, not a throughput
one, and A100 deployments usually care about throughput.

The most honest summary: the *method* transfers, the *numbers* do not, and the
optimal K would have to be re-measured.

**Q: Isn't the draft model just wasted compute when it's wrong?**

Yes, and that is exactly the trade. It is a bet with bounded downside — at worst
K·c extra work per emitted token — and a payoff of up to K+1 tokens per target
pass. The bet is worth taking whenever α > 1 + K·c. The measured sweep is the
empirical test of that inequality across K.

**Q: How do you know your implementation is correct and not just fast?**

Three independent lines of evidence, and it is worth saying why one alone would
not be enough. Greedy equivalence catches indexing errors but says nothing about
the sampler, since at temperature 0 the rejection rule degenerates. The
synthetic Monte Carlo tests catch sampler errors but use no models, so they say
nothing about indexing or cache handling. The rollback tests catch stale-cache
bugs that neither of the others would surface, because a stale cache produces
fluent, plausible, wrong text. Together they cover the three places this
algorithm actually breaks.

**Q: Your speedup is around 1.3×. The papers claim 2–3×. Why the gap?**

Four honest reasons, none of which is "the implementation is wrong".

*The drafter is weak and not same-family.* llama-68m was not distilled from
TinyLlama; it is simply a small Llama-architecture model. Acceptance is
Σ min(p, q) = 1 − TVD(p, q), so a drafter that disagrees with the target often
has a low β by construction. This is precisely the gap the EAGLE head exists to
close.

*Both models are small, so overheads dominate more.* The measured cost of one
target forward pass is about 87 ms, against a bandwidth-roofline estimate of
11.5 ms — the stack achieves roughly 13% of the roofline. Kernel launch latency,
Python, and WDDM overhead are a large constant that a 1.1 B model cannot
amortise the way a 70 B model does. Papers reporting 2–3× generally use targets
where the weight read genuinely dominates.

*Thermally limited laptop hardware*, and a paired interleaved design with
confidence intervals rather than a best-of-N point estimate. A best-of-N number
would be higher and would mean less.

The strongest thing to say, though, is that **the model of the system predicts
its own behaviour**. With the measured c = 0.069, α/(1 + K·c) gives 1.39 at
K = 4 against a measured 1.30; the residual is the extra cost of the K+1-token
block over a single-token block plus per-round Python. Predicting the observed
speedup from independently measured α and c, and being able to say where the
remaining discrepancy comes from, is a better answer than a bigger number.

**Q: Tell me about a bug you found in this project and how you found it.**

The EAGLE head trained to a cross-entropy of 0.053, which looked excellent. The
check that caught it was noticing the loss had a *known lower bound*: the head
is distilling the target's own next-token distribution, so it cannot beat the
target's cross-entropy on the same data. Measuring that floor took a few lines —
decode the cached features through the frozen LM head, score against the stored
next tokens — and gave 2.56. A loss roughly fifty times below the teacher's is
not learning; the correctly-trained head lands at 2.60, just above the floor.

The cause was `attention_mask=None` reaching a `LlamaDecoderLayer` on the eager
attention path, which applies no mask at all; attention was bidirectional and
the head was reading the label off position t+1. Section 4.3.1 has the full
account.

Two things worth drawing out. First, the bug's only symptom would have been
"EAGLE gives no speedup" — a plausible, well-supported, wrong conclusion, not an
obvious failure. Second, the fix is not the interesting part; the reusable part
is that **any distillation loss has a floor, and checking against it is nearly
free**. That check now runs automatically, and the sweep refuses to benchmark a
head that trips it.

**Q: Your EAGLE head is slower than the baseline. Doesn't that mean EAGLE
doesn't work?**

No — it means my head was undertrained, and I can show that rather than assert
it. Two pieces of evidence.

First, the generalisation gap. Measured on cached documents from its own
training corpus, the head's acceptance probability is about 0.28. Measured on
the held-out benchmark prompts it is about 0.05 — that is what an acceptance
length of 1.047 at K=1 implies, and K=1 is the clean case because only the
first draft step runs, on a real target feature. Same head, same temperature,
roughly a six-fold gap. It memorised 250,000 tokens rather than learning the
target's distribution.

Second, the capacity sweep. Larger heads reached lower *training* cross-entropy
— 2.00 at 96.5 M parameters against 2.60 at 35.1 M — while benchmark acceptance
stayed flat at 1.06 to 1.09 tokens per round. Capacity bought fit and bought no
generalisation. That is overfitting's signature, and it is not what a method
failing on its merits looks like.

There is also a cost half that has nothing to do with training. The head is
about twice as expensive per draft step as llama-68m — c of 0.11 to 0.16
against 0.064 to 0.088 — because it projects from hidden 2048 through a
32,000-row vocabulary in FP32, while llama-68m projects from hidden 768 in
FP16. So it lost on both terms of alpha / (1 + K*c) at once. Running the head in
FP16 would roughly halve that term and is the first thing I would fix.

The part I would want credit for is that the same two-parameter cost model that
predicts the Phase 1 sweep also predicts every EAGLE point to within one to two
percent. The negative result is not an anomaly; it is the model working.

**Q: How do you know that isn't just a bug in your EAGLE implementation?**

Because three bugs in it were found and fixed before these numbers were taken,
and each one is now covered by a test that fails without the fix.

The head's attention was bidirectional during training, so it read labels from
the future — caught by the loss falling roughly fifty times below the teacher's
cross-entropy floor. The inference feature/token pairing was off by one relative
to training. And the head ran without a KV cache at inference, a computation it
was never trained for.

Worth being honest about the third one: I predicted fixing it would raise
acceptance from about 1.13 to about 1.35 tokens per round. It did not — the
corrected path measured 1.05 to 1.06, statistically indistinguishable from the
broken one. The fix was necessary for correctness and irrelevant to the outcome,
and finding that out is what redirected the diagnosis to overfitting, which is
where the effect actually was.

Beyond that, the structural argument: in speculative decoding every draft-side
bug is a performance bug, because the accept/reject rule absorbs it. So the
draft path cannot be validated by output correctness and needs exactly the kind
of targeted tests above.

**Q: What would you do next, with more time or better hardware?**

In priority order, and the order is set by what the measurements say rather than
by what sounds interesting.

**Train the head on enough data.** This is the whole Phase 2 result. The
measured generalisation gap — acceptance 0.28 on training documents against 0.05
held out — says the head is data-starved, not that the architecture is wrong.
Collection is the bottleneck: features cost 4 KB per token on disk and the
target has to run over the corpus once. Scaling to tens of millions of tokens
means streaming the collection rather than caching all of it, which is a real
engineering change, not a bigger number in a config.

**Run the head in FP16.** Roughly halves c, which the cost model says converts
directly into speedup. Cheap, and worth doing before any conclusion about
whether feature-level drafting is competitive here.

**Hold out a validation split during head training.** The overfitting was only
visible because the benchmark disagreed with the training loss. A validation
split would have shown it during training, in minutes, instead of after a
two-hour sweep.

**Tree attention**, so several draft candidates are verified in one pass rather
than a single linear chain — raises α without raising the number of target
passes, and is the main thing separating this implementation from a modern one.

**Adapt K online** from a running acceptance estimate. The measured optimum
differs between greedy (K=2) and sampling (K=3), and by prompt domain, so a
fixed K is leaving speedup on the table.

**Measure at longer contexts and non-unit batch sizes**, to find where the
bandwidth-bound premise that justifies the whole method stops holding.
