"""Loss functions for AgentJev supervision types.

Every question carries a supervision type and a confidence weight:
  - known_distribution / deterministic: weight 1.0, soft cross-entropy on
    the gold distribution
  - binomial_counts: k successes in n trials (boolean questions, exactly
    two candidates, candidate 0 = TRUE), binomial negative log-likelihood
    per trial, weight scales with trials
  - multiclass_counts: count vector over candidates, multinomial NLL per
    trial, weight scales with trials
  - empirical / heuristic / teacher: soft cross-entropy, lower weight
    (empirical by trial count, heuristic ~0.2, teacher 0.1-0.3)

Main-loss aggregation is per-question over ALL questions in the batch:
  L_main = sum_q(w_q * ell_q) / sum_q(w_q)
so splitting a batch into microbatches and recombining (with weight sums
accumulated as numerators/denominators) gives consistent gradients,
unlike the previous per-supervision-type weighted means.

Auxiliary terms: Brier score, pairwise margin ranking (default OFF,
weight 0 in configs; symmetric over all strictly-ordered pairs),
ordinal CDF loss, permutation-invariance KL consistency.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SUP_CODES = {
    "known_distribution": 0,
    "deterministic": 0,
    "binomial_counts": 1,
    "multiclass_counts": 2,
    "empirical": 3,
    "heuristic": 3,
    "teacher": 3,
}


def masked_log_softmax(logits: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """log-softmax over valid candidates. Masked positions get 0.0 (not
    -inf) so that products with zero targets do not produce NaN."""
    logits = logits.float().masked_fill(~cand_mask, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    return torch.where(cand_mask, logp, torch.zeros_like(logp))


def _wmean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def soft_ce_per_q(logits, target_dist, cand_mask) -> torch.Tensor:
    logp = masked_log_softmax(logits, cand_mask)
    return -(target_dist * logp).sum(-1)


def soft_cross_entropy(logits, target_dist, cand_mask, weight):
    return _wmean(soft_ce_per_q(logits, target_dist, cand_mask), weight)


def binomial_nll_per_q(logits, counts, trials, cand_mask) -> torch.Tensor:
    # Exactly two candidates; candidate 0 = TRUE (success), 1 = FALSE.
    logp = masked_log_softmax(logits, cand_mask)
    k = counts[:, 0]
    n = trials  # caller filters trials > 0
    return -(k * logp[:, 0] + (n - k) * logp[:, 1]) / n


def binomial_count_loss(logits, counts, trials, cand_mask, weight):
    return _wmean(binomial_nll_per_q(logits, counts, trials, cand_mask), weight)


def multiclass_nll_per_q(logits, counts, trials, cand_mask) -> torch.Tensor:
    logp = masked_log_softmax(logits, cand_mask)
    return -(counts * logp).sum(-1) / trials


def multiclass_count_loss(logits, counts, trials, cand_mask, weight):
    return _wmean(multiclass_nll_per_q(logits, counts, trials, cand_mask), weight)


def masked_softmax(logits: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    return masked_log_softmax(logits, cand_mask).exp() * cand_mask


def brier_loss(logits, target_dist, cand_mask, weight):
    p = masked_softmax(logits, cand_mask)
    per_q = ((p - target_dist).pow(2) * cand_mask).sum(-1)
    return _wmean(per_q, weight)


def permutation_invariance_kl(logits, logits_perm, cand_mask, weight):
    # Symmetrized KL between the distribution from the original candidate
    # order and from a permuted order (aligned back). ~0 when the head is
    # exactly equivariant.
    logp = masked_log_softmax(logits, cand_mask)
    logq = masked_log_softmax(logits_perm, cand_mask)
    p, q = masked_softmax(logits, cand_mask), masked_softmax(logits_perm, cand_mask)
    kl_pq = (p * (logp - logq)).sum(-1)
    kl_qp = (q * (logq - logp)).sum(-1)
    return _wmean(0.5 * (kl_pq + kl_qp), weight)


def pairwise_margin_loss(logits, target_dist, cand_mask, weight, margin=0.5):
    # Symmetric over ALL strictly-ordered pairs (t_i > t_j): pull logit i
    # above logit j by `margin`. No dependence on argmax tie-breaking.
    # NOTE: with nonzero weight this shifts the joint optimum away from
    # the calibrated distribution; keep weight 0 for calibration training.
    t = target_dist
    li = logits.unsqueeze(2)  # [Bq, C, 1]
    lj = logits.unsqueeze(1)  # [Bq, 1, C]
    ti = t.unsqueeze(2)
    tj = t.unsqueeze(1)
    pair = cand_mask.unsqueeze(2) & cand_mask.unsqueeze(1) & (ti > tj + 1e-8)
    viol = F.softplus(margin - (li - lj)) * pair
    n_pair = pair.sum(dim=(1, 2))
    per_q = viol.sum(dim=(1, 2)) / n_pair.clamp_min(1).to(viol.dtype)
    has_pair = (n_pair > 0).to(weight.dtype)
    return _wmean(per_q, weight * has_pair)


def ordinal_cdf_loss(logits, target_dist, cand_mask, weight):
    # Candidates are ordered levels; match the full CDF, excluding the
    # final level (CDF is always 1 there). Caller filters rows with <2
    # valid candidates.
    p = masked_softmax(logits, cand_mask)
    cdf_p = p.cumsum(-1)[:, :-1].clamp(1e-6, 1 - 1e-6)
    cdf_t = target_dist.cumsum(-1)[:, :-1]
    level_mask = cand_mask[:, 1:]  # threshold level exists
    bce = -(cdf_t * cdf_p.log() + (1 - cdf_t) * (1 - cdf_p).log())
    per_q = (bce * level_mask).sum(-1) / level_mask.sum(-1).clamp_min(1)
    return _wmean(per_q, weight)


def compute_losses(outputs: dict, batch: dict, w_cfg: dict,
                   margin: float = 0.5) -> dict:
    """Dispatch per-question losses by supervision type.

    The main loss is a single per-question weighted mean over ALL
    questions (numerator = sum of w_q * ell_q across supervision types,
    denominator = sum of all w_q in the batch). Per-type means are still
    returned under ``ce`` / ``binomial`` / ``multiclass`` for logging.

    Returns a dict of scalar tensors, including ``total``.
    """
    logits, cand_mask = outputs["logits"], outputs["cand_mask"]
    sup = batch["sup_type"]
    weight = batch["weight"]
    t = batch["target_dist"]
    counts, trials = batch["counts"], batch["trials"]

    m_dist = (sup == SUP_CODES["known_distribution"]) | (sup == SUP_CODES["empirical"])
    # binomial: exactly 2 candidates and at least one trial, else skip
    m_bin = (sup == SUP_CODES["binomial_counts"]) & (trials > 0) \
        & (cand_mask.sum(-1) == 2)
    m_multi = (sup == SUP_CODES["multiclass_counts"]) & (trials > 0)

    numerator = logits.sum() * 0.0  # keeps graph connectivity on empty subsets
    terms = {}
    if m_dist.any():
        ce_q = soft_ce_per_q(logits[m_dist], t[m_dist], cand_mask[m_dist])
        numerator = numerator + (ce_q * weight[m_dist]).sum()
        terms["ce"] = _wmean(ce_q, weight[m_dist])
    if m_bin.any():
        bin_q = binomial_nll_per_q(logits[m_bin], counts[m_bin],
                                   trials[m_bin], cand_mask[m_bin])
        numerator = numerator + (bin_q * weight[m_bin]).sum()
        terms["binomial"] = _wmean(bin_q, weight[m_bin])
    if m_multi.any():
        mul_q = multiclass_nll_per_q(logits[m_multi], counts[m_multi],
                                     trials[m_multi], cand_mask[m_multi])
        numerator = numerator + (mul_q * weight[m_multi]).sum()
        terms["multiclass"] = _wmean(mul_q, weight[m_multi])

    terms["main"] = numerator / weight.sum().clamp_min(1e-8)

    # Auxiliary terms over their applicable subsets.
    if w_cfg.get("brier", 0.0) > 0:
        terms["brier"] = brier_loss(logits, t, cand_mask, weight)
    if w_cfg.get("margin", 0.0) > 0 and m_dist.any():
        terms["margin"] = pairwise_margin_loss(logits[m_dist], t[m_dist],
                                               cand_mask[m_dist], weight[m_dist],
                                               margin=margin)
    m_ord = batch["ordinal"] & (cand_mask.sum(-1) > 1)
    if w_cfg.get("ordinal", 0.0) > 0 and m_ord.any():
        terms["ordinal"] = ordinal_cdf_loss(logits[m_ord], t[m_ord],
                                            cand_mask[m_ord], weight[m_ord])
    if w_cfg.get("perm_kl", 0.0) > 0 and outputs.get("logits_perm") is not None:
        terms["perm_kl"] = permutation_invariance_kl(logits, outputs["logits_perm"],
                                                     cand_mask, weight)

    total = terms["main"]
    for name, val in terms.items():
        if name in ("main", "ce", "binomial", "multiclass"):
            continue  # ce/binomial/multiclass are logging diagnostics of main
        total = total + val * w_cfg.get(name, 0.0)
    terms["total"] = total
    return terms
