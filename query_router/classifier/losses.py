"""Cost-sensitive ordinal-aware loss — DESIGN.md §6.2 decision #2.

Plain cross-entropy only looks at p(true_class), so it can't distinguish a
tier-2-as-tier-3 miss from a tier-0-as-tier-3 miss. This adds a term that
penalizes probability mass in proportion to its distance from the true tier,
directly targeting the BRD's "adjacent accuracy" requirement (never off by
more than one tier).

The distance penalty is asymmetric: over_penalty > under_penalty by default,
so predicting a tier ABOVE the true one (routing to a costlier model than
needed) is punished harder than predicting below it. Higher tiers always
cost more, so an unnecessary escalation is the expensive mistake to avoid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CostSensitiveCrossEntropy(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        lambda_dist: float = 0.5,
        over_penalty: float = 1.5,
        under_penalty: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dist = lambda_dist
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, targets, reduction="none")

        classes = torch.arange(self.num_classes, device=logits.device).unsqueeze(0)
        signed_dist = classes - targets.unsqueeze(1)  # >0 = predicted tier above truth (escalation)
        weight = torch.where(signed_dist > 0, self.over_penalty, self.under_penalty)
        distance_penalty = (signed_dist.abs().float() * weight * probs).sum(dim=1)

        return (ce + self.lambda_dist * distance_penalty).mean()
