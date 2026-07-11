# Why Imbalanced Losses

## The failure mode of standard cross-entropy

Cross-entropy and BCE minimize the expected log-loss over the training distribution. In principle nothing is wrong with that: BCE is a proper scoring rule, and an unconstrained model minimizing it would recover the true conditional probabilities. The failure is in the fit a real model can reach. With shared features and limited capacity, when the training distribution is heavily skewed — for example, 99% negative examples in fraud detection — the model that minimizes *average* log-loss trades minority-class accuracy for majority-class accuracy, because errors on the majority cost far more total loss.

Even with good regularization, gradient updates from 990 negative examples per 10 positives tend to push the model toward predicting negative. The result is high accuracy (correctly predicting "negative" 99% of the time) but near-zero precision and recall on the class that matters.

## Why focal loss helps for imbalanced classification

Focal loss (Lin et al., ICCV 2017) modifies cross-entropy by multiplying each sample's loss by `(1 - p_t)^gamma`, where `p_t` is the model's confidence in the correct label:

- Samples the model already classifies correctly (high `p_t`) get multiplied by a small factor, reducing their contribution
- Hard examples (low `p_t`) keep their full loss contribution

This re-weighting means easy negatives — which dominate the loss under imbalance — are down-weighted while hard positives retain full gradient signal. The `alpha` parameter additionally re-weights the positive class globally.

Focal loss is best thought of as a better BCE/CE, not a fundamentally different objective. It still minimizes a surrogate of the 0/1 loss, not a ranking metric.

## Why ranking losses for ranking metrics

Average Precision (AP) and Recall at a threshold are ranking metrics: they depend on how samples are ordered by score, not on absolute probabilities. Cross-entropy is a poor surrogate for ranking because a model can achieve high AP while having poorly calibrated probabilities, and vice versa.

`SmoothAPLoss` directly approximates AP by estimating soft ranks via sigmoid differences. Gradients push positives to rank above negatives, which is exactly the objective AP measures.

`RecallAtQuantileLoss` targets a different operating point: it asks "what fraction of positives score in the top q% of all scores?" This is natural for alert/detection settings where only a fixed number of alerts can be reviewed.

`PAUCAtBudgetLoss` sits between these two: it optimizes a normalized partial AUC over a false-positive-rate band `[alpha, beta]` that brackets a target operating point (e.g. FPR ≈ 0.005). Where `SmoothAPLoss` integrates across the full curve and `RecallAtQuantileLoss` pins a single threshold, `PAUCAtBudgetLoss` optimizes a *band* of the ROC. This is the right choice when the business constraint is a fixed false-alarm budget and you want the model to be robust across a small neighborhood of that operating point, not just at one exact score cutoff.

## When to use each loss

| Situation | Recommended loss |
|---|---|
| Mild imbalance, want drop-in replacement | `SigmoidFocalLoss` or `SoftmaxFocalLoss` |
| Binary / multi-label classification | `SigmoidFocalLoss` |
| Multiclass with rare classes | `SoftmaxFocalLoss` with `alpha` |
| Detection with many background anchors | `SoftmaxFocalLoss` with `mean_positive` |
| Optimizing AUCPR directly | `SmoothAPLoss` |
| Fixed false-alarm budget / band around an operating point | `PAUCAtBudgetLoss` |
| Fixed operating point (top N% flagged) | `RecallAtQuantileLoss` |
| Early unstable gradients from ranking loss | `LossWarmupWrapper` with warmup on CE/BCE |

## Assumptions and failure modes

Each loss makes assumptions about the data, the model, and the training regime. Understanding when those assumptions break helps you decide whether a given loss is appropriate for your problem. See [Assumptions and Failure Modes](assumptions-and-failure-modes.md) for a detailed per-loss analysis.

## Trade-offs

**Focal loss** is fast (O(N)), requires no queue, and is a drop-in replacement. It does not directly optimize AP.

**Ranking losses** directly optimize the metric of interest but require a minimum pool size to produce stable rank estimates. At very low positive rates (< 1%), without a memory queue the loss degenerates because a single batch may contain zero positives. The queue solves this; the core computation is O(|P| × M) where |P| is the positive count and M is the pool size. `PAUCAtBudgetLoss` additionally requires the pooled iid-negative count to comfortably resolve the band's smaller nonzero edge for the band-edge quantiles to be unbiased: substantially more than `1/beta` iid negatives with the default `alpha=0` band, and more than `1/alpha` only when `alpha > 0` — the same flavor of requirement as `RecallAtQuantileLoss`'s `(M × q) ≥ 10` guideline.

**LossWarmupWrapper** adds complexity but addresses the cold-start problem: ranking loss gradients are uninformative when model scores are near-uniform. BCE warmup first trains the model into a reasonable score space before switching.
