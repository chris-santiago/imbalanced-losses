# Assumptions and Failure Modes

This page answers the question: *"Is my problem a good fit for these losses?"*

Each loss makes implicit and explicit assumptions about the data, the model, and the training regime. When those assumptions hold, the loss functions as intended. When they break, the loss produces misleading gradients, degenerate solutions, or silent failures.

---

## Universal assumptions

All losses in this library share these baseline requirements. Violations affect every loss, not just one.

### Label quality

All losses assume that positive labels are correct. Focal loss and ranking losses amplify the influence of hard examples — but a hard example is indistinguishable from a mislabeled one. Label noise in the positive class is especially dangerous: a mislabeled negative scored low will be treated as a hard positive and receive full gradient weight.

**Rule of thumb:** Positive-class label error rates above ~5% tend to degrade focal loss; ranking losses are even more sensitive because a single mislabeled positive shifts the rank estimate for the entire pool.

### Sufficient model capacity

Losses cannot compensate for a model that cannot discriminate between classes. If the feature space does not separate positives from negatives, no loss function fixes that. These losses are designed to better *direct* the gradient signal, not to *create* signal where none exists.

### Score distribution stability within the queue window

The memory queue assumes the model's score distribution changes slowly relative to the queue rotation period (`queue_size / batch_size` steps). If the distribution shifts dramatically within that window — due to curriculum learning, staged unfreezing, or learning rate spikes — stale queue entries become misleading. Reset the queue manually after any such event.

---

## Focal Loss

`SigmoidFocalLoss` and `SoftmaxFocalLoss` are modified cross-entropy losses. They inherit CE's theoretical framework and fail in the same ways CE fails, plus some additional failure modes specific to the focal modifier.

### When it works

- Moderate to severe imbalance: the original RetinaNet paper (Lin et al., ICCV 2017) used 1:1000 foreground-to-background ratios. Focal loss was specifically designed for this regime.
- Easy examples dominate gradient noise: the $(1 - p_t)^\gamma$ modifier is most valuable when the majority class is easy and the minority class is hard.
- You need a drop-in CE replacement with no queue or infrastructure changes.
- Mild imbalance where tuning `alpha` alone is insufficient.

### When it breaks down

**Extreme imbalance (< 0.01% positive rate)**

At very low positive rates, even after down-weighting easy negatives, the absolute number of positive gradient contributions per batch is near zero. Focal loss re-weights contributions but does not create positives that aren't there. You still need enough positives per batch to learn. At 0.01% with batch size 64, expect < 1 positive per batch.

**Label noise amplified by high gamma**

As $\gamma$ increases, down-weighting of correctly-classified easy examples becomes more aggressive. A mislabeled negative (true negative, labeled positive) that the model correctly assigns a low score to is treated as a hard positive and receives a large gradient weight. Empirically, $\gamma > 3$ tends to increase sensitivity to label noise. Lin et al. found $\gamma = 2$ optimal across a range of detection benchmarks; values above 3 showed diminishing returns and increasing instability.

**Optimizing for a ranking metric**

Focal loss minimizes a weighted log-loss, which is a proxy for calibrated probability estimation. It does *not* directly optimize Average Precision, AUROC, or recall at a threshold. A model trained with focal loss may have better AP than one trained with CE — but this is an indirect effect, not a guarantee. If your evaluation metric is AP or recall@k, `SmoothAPLoss` or `RecallAtQuantileLoss` will generally outperform focal loss at comparable imbalance levels.

**Calibration is required downstream**

Focal loss distorts the predicted probability distribution. The down-weighting of easy examples pushes the model toward underconfident predictions on the majority class and overconfident predictions on hard examples. Do not use focal loss output probabilities as calibrated estimates without a calibration step (Platt scaling, isotonic regression, etc.).

**alpha/gamma interaction with severely imbalanced multi-class**

`SoftmaxFocalLoss` with per-class `alpha` can produce unstable training when rare classes have very few samples. The model can "ignore" a rare class by assigning its alpha weight to a region of parameter space that is rarely visited — effectively the rare class receives no gradient updates across many batches. If a class appears fewer than once per ~20 batches, consider either oversampling or switching to `SmoothAPLoss`.

---

## SmoothAPLoss

This loss directly approximates Average Precision using a sigmoid-based soft rank estimator (Brown et al., ECCV 2020). The approximation is theoretically sound, but its quality depends on pool size, temperature, and positive rate.

### When it works

- Direct optimization of AUCPR / Average Precision is the goal.
- Positive rate is in the range 0.1%–20%. Below this range, the queue must be large enough to accumulate sufficient positives; above this range, focal loss is likely competitive with much lower overhead.
- The pool (batch + queue) reliably contains ≥ 10–20 positives. This is the practical threshold for stable AP estimation. With 10 positives, the AP estimate has high variance but usable gradient signal; with < 5, the estimate is essentially noise.
- Score ranges across the pool are comparable — no extreme outliers that compress all other soft ranks to near 0 or 1.

### When it breaks down

**Pool too small for the positive rate**

If `batch_size + queue_size` is too small to accumulate enough positives, the soft AP estimate is highly variable and training can oscillate. At a 0.5% positive rate, you need M ≈ 2000–4000 to reliably see 10–20 positives per step. If M is limited by memory, consider `RecallAtQuantileLoss`, which does not require counting positives globally.

**Pool too large for memory (seq2seq)**

After flattening a seq2seq batch from `[B, T, C]` to `[B*T, C]`, the pool size M = B × T can be very large (e.g. 30 × 512 = 15 360). The pairwise matrix `[|P|, M]` in the soft rank computation, retained for all C classes simultaneously in the autograd graph, causes O(M²) peak memory and will OOM even with `queue_size=0` and a reduced batch size.

Use `max_pool_size` to cap the pool with stratified random subsampling:

```python
loss_fn = SmoothAPLoss(num_classes=C, queue_size=1024, max_pool_size=4096)
```

The queue accumulates the original full batch (unaffected by the cap). A one-time `UserWarning` fires when subsampling first triggers. Because the subsampled pool is a random subset, the loss value varies across steps for identical inputs — this is expected and analogous to dropout noise.

**Temperature too low in early training**

When model scores are near-uniform (random initialization), all pairwise score differences are close to zero. At low temperature τ, $\sigma(\Delta s / \tau)$ saturates near 0.5 everywhere — soft ranks are all approximately $M/2$ and the loss is approximately 0.5 regardless of the model's output. Gradients nearly vanish. This is why cold-starting with focal loss via `LossWarmupWrapper` is recommended: it shapes the score space before AP loss is activated.

**Temperature too low for gradient variance**

Even in mid-training, very low τ (< 0.005) can produce gradients that are highly sensitive to small score perturbations. When a positive score crosses a negative score, the soft rank changes sharply, producing large gradient spikes. In practice this can manifest as sudden loss spikes or training instability. Use the geometric decay schedule in `LossWarmupWrapper` to approach low temperatures gradually.

**All positives or all negatives in the pool**

AP is undefined when the pool contains no positives or only positives. The loss returns 0.0 for empty pools (no positives) and marks the class as invalid (NaN for `reduction='none'`). This is correct behavior, but if your batches systematically produce degenerate pools — e.g., a class so rare that even with a full queue it never appears — the loss never trains that class. Monitor per-class positive rates and ensure queue size is sufficient.

**Queue staleness after distribution shift**

Queue entries are detached and treated as a fixed reference distribution for the current step. This is valid when the model's score distribution evolves slowly. After events that shift the distribution abruptly — phase switches, checkpoint loading, learning rate resets — the queue contains entries that misrepresent the current model's output range. The soft ranks computed against a stale queue are biased. Reset the queue after any such event.

**Multi-modal score distributions**

The soft rank formula sums sigmoid comparisons uniformly across the pool. If the score distribution is bimodal (e.g., two well-separated clusters of negatives), positives caught between the modes receive a very different rank signal than positives in the upper cluster. This is not a failure mode per se, but it means the loss is sensitive to the overall score distribution shape. Monitoring the raw score histogram during training is helpful.

**Theoretical note on the Brown et al. approximation**

The ECCV 2020 paper demonstrated that the sigmoid soft rank converges to the true AP as τ → 0, and that directly optimizing this surrogate outperforms post-hoc tuning of cross-entropy on standard retrieval benchmarks. However, the paper evaluated on retrieval tasks (where the pool is the entire gallery set) rather than classification with a small batch + queue. In the small-pool regime, the approximation quality is lower and the variance of the gradient estimator is higher than in the original paper's setting.

---

## RecallAtQuantileLoss

This loss optimizes recall above a score threshold set at the q-th quantile of the pooled score distribution. It is an original design (not from published literature), motivated by fixed-capacity review settings.

### When it works

- You operate at a fixed precision point: only the top q-fraction of flagged items will be reviewed.
- Quantile q > positive class fraction: this ensures the threshold θ typically falls in the negative score region under a well-trained model. If q = 0.01 and your positive rate is 2%, the threshold must be in the negative region for recall to be maximized — this is the natural operating regime.
- The pool is large enough to estimate the quantile reliably. At q = 0.005, you need at least 1/0.005 = 200 pooled samples for the quantile estimate to correspond to at least one sample. In practice, 5–10× more samples (1000–2000) give a stable estimate.
- Positive scores are dispersed above the threshold: gradients flow from positives that score below θ, pushing them above it. If all positives already score well above θ, the loss is near zero and training stalls (correctly — the objective is already met).

### When it breaks down

**Quantile < positive class fraction**

If q < positive_rate, then under a perfect model (all positives score above all negatives), the quantile threshold falls inside the positive score range. Positives above θ contribute zero gradient (they are already above the threshold), but positives below θ receive push-up gradients toward a threshold that is already inside the positive cluster. The loss can converge to a state where ~q fraction of positives are above threshold and the rest are not — a partial solution that is locally stable.

**Stop-gradient on the threshold**

The quantile threshold θ is computed with `detach()` — no gradient flows through it. This is intentional: allowing gradients through θ would incentivize the model to push *all* scores below the threshold (to lower the threshold and trivially satisfy the objective). However, the stop-gradient means the loss has no signal if *all* positives already score above θ at a given step. In that case the per-positive sigmoids are all near 1.0, the loss is near 0.0, and gradients vanish — even if the threshold is poorly positioned. This is correct when recall is actually high, but it means the loss cannot push θ lower.

**Threshold instability at score distribution boundaries**

If the score distribution has a gap or discontinuity at the quantile, the threshold can jump between steps. Since the threshold is recomputed fresh each forward pass from the current pool, a score jump of even a small amount can cause θ to shift by a large amount if the distribution is sparse near the q-th percentile. This produces inconsistent gradient signals across consecutive steps. Adding queue entries stabilizes this by densifying the distribution.

**Pool too small for the target quantile**

Unlike `SmoothAPLoss`, `RecallAtQuantileLoss` requires enough samples to estimate a specific percentile. At q = 0.005 with a pool of 100, the quantile is determined by the single lowest-scoring sample in the top 0.5% — just one data point. This estimate is highly variable. The queue size should be set so that `(batch_size + queue_size) * q ≥ 10` for a stable threshold estimate.

**Sensitivity to score scale**

The gradient magnitude of the per-positive sigmoid $\sigma((s_i - \theta) / \tau)$ depends on $(s_i - \theta) / \tau$. If the model's scores have large scale (e.g., logits in the range ±50 rather than ±5), the gradient is nearly zero for all positives not in an infinitesimally thin band around θ. This produces a very sparse and spiky gradient signal. Normalize logits or adjust τ to match the expected score scale.

---

## LossWarmupWrapper

This utility is not a loss itself but manages the transition from a warmup loss to the main ranking loss. Its failure modes are training dynamics failures, not mathematical ones.

### When it works

- The model starts from random initialization or a weakly-supervised checkpoint where scores are near-uniform.
- The warmup loss (CE/BCE) produces a meaningful score ordering before the AP phase begins.
- The blend and temperature schedules are appropriate for the total training budget.

### When it breaks down

**Warmup phase too short**

If the model hasn't developed a meaningful score ordering by the end of warmup, the AP loss starts from effectively random scores — the same cold-start problem it was designed to avoid. Scores near-uniform at the start of AP phase produce near-zero gradients (see temperature discussion above). Rule of thumb: warmup until the model achieves at least moderate AP (> 0.3 on a mid-difficulty task) before switching.

**Warmup phase too long**

Prolonged BCE warmup can cause the model to overfit to a calibrated-probability objective. The model learns to predict the exact positive rate rather than to *rank* positives above negatives. When the AP loss is then activated, the score distribution may be well-calibrated but not discriminative — positives and negatives are separated only modestly. This can be detected by monitoring AUCPR during warmup: if it plateaus early, the warmup phase can be shortened.

**Using with a pretrained model**

If the model is initialized from a pretrained checkpoint that already produces meaningful scores, the warmup phase may be unnecessary and can even harm training by pulling the model away from a good initialization toward a worse one. In this case, consider skipping the warmup phase entirely (`warmup_epochs=0`) or using a very short warmup (1–2 epochs) with a high learning rate decay.

**Temperature decay too fast**

If `temp_end / temp_start` is too small or `temp_decay_steps` is too short, the temperature reaches the minimum before the model has refined the ranking. Low temperature with poorly separated scores produces large, noisy gradients. Schedule the decay to reach `temp_end` no earlier than ~50–75% of the main training phase.

**Queue poisoning at the phase switch**

`LossWarmupWrapper` automatically resets the queue at the phase switch. However, if you manually bypass the wrapper (e.g., by directly calling `main_loss.forward()` during warmup), the queue accumulates warmup-era logits that are not reset. Always use the wrapper's `forward()` method to ensure correct queue management.

---

## Diagnostic summary

The table below maps common failure symptoms to root causes and remedies.

| Symptom | Most likely cause | Remedy |
|---|---|---|
| Loss stays near 0.5 from the start | Temperature too low for uniform scores | Use `LossWarmupWrapper`; start with high temperature |
| Loss oscillates wildly | Temperature too low for current score scale | Increase temperature; check score range |
| Rare class never improves | Pool contains zero positives for that class | Increase queue size; check per-class positive rate |
| AP loss worse than CE | Cold start: scores too uniform when AP phase begins | Lengthen warmup; use `LossWarmupWrapper` |
| RecallAtQuantile stalls | All positives already above threshold | Normal convergence; also verify quantile setting |
| RecallAtQuantile unstable threshold | Pool too sparse near quantile | Increase queue size; check `(M * q) ≥ 10` |
| Focal loss noisy despite tuning | High label noise in positive class | Reduce γ; inspect label quality |
| DDP: loss diverges across workers | Missing all-gather | Set `gather_distributed=True` or use auto-detect |
| Training spike after phase switch | Queue poisoning from warmup logits | Verify `LossWarmupWrapper` handles the switch; check queue reset |
| Loss near zero but metrics poor | Score scale mismatch with temperature | Normalize logits or scale τ to match score range |
