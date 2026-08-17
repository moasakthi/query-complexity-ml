import torch
import torch.nn.functional as F

from query_router.classifier.losses import CostSensitiveCrossEntropy


def test_far_miss_penalized_more_than_adjacent_miss():
    loss_fn = CostSensitiveCrossEntropy(num_classes=4, lambda_dist=1.0)

    # True label is tier 0. One logits vector puts most mass on tier 1 (adjacent
    # miss), the other puts the same amount of mass on tier 3 (far miss) — same
    # "wrongness" under plain cross-entropy, but the far miss should cost more here.
    target = torch.tensor([0])
    adjacent_logits = torch.tensor([[0.1, 5.0, 0.1, 0.1]])
    far_logits = torch.tensor([[0.1, 0.1, 0.1, 5.0]])

    assert loss_fn(far_logits, target) > loss_fn(adjacent_logits, target)


def test_reduces_to_plain_ce_when_lambda_zero():
    loss_fn = CostSensitiveCrossEntropy(num_classes=4, lambda_dist=0.0)
    logits = torch.randn(5, 4)
    targets = torch.randint(0, 4, (5,))
    assert torch.allclose(loss_fn(logits, targets), F.cross_entropy(logits, targets))


def test_over_escalation_penalized_more_than_under_escalation():
    loss_fn = CostSensitiveCrossEntropy(num_classes=4, lambda_dist=1.0, over_penalty=1.5, under_penalty=1.0)

    # True label is tier 2. Equal-magnitude mass placed one tier above (costly
    # over-escalation) vs one tier below (cheap under-escalation) truth.
    target = torch.tensor([2])
    over_logits = torch.tensor([[0.1, 0.1, 0.1, 5.0]])   # mass on tier 3
    under_logits = torch.tensor([[0.1, 5.0, 0.1, 0.1]])  # mass on tier 1

    assert loss_fn(over_logits, target) > loss_fn(under_logits, target)
