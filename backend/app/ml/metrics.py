from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None

def _ensure_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for training metrics. "
            "Install the packages from backend/requirements.txt before running the notebooks."
        )


def dice_coefficient_from_logits(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    _ensure_torch()
    predictions = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    cardinality = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    score = (2.0 * intersection + eps) / (cardinality + eps)
    return float(score.mean().item())


def iou_from_logits(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    _ensure_torch()
    predictions = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    score = (intersection + eps) / (union + eps)
    return float(score.mean().item())


def pixel_accuracy_from_logits(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    threshold: float = 0.5,
) -> float:
    _ensure_torch()
    predictions = (torch.sigmoid(logits) >= threshold).float()
    correct = (predictions == targets.float()).float().mean()
    return float(correct.item())


def attribute_accuracy(
    outputs: dict[str, "torch.Tensor"],
    targets: dict[str, "torch.Tensor"],
) -> dict[str, float]:
    _ensure_torch()
    scores: dict[str, float] = {}
    for field in outputs.keys():
        predictions = outputs[field].argmax(dim=1)
        scores[field] = float((predictions == targets[field]).float().mean().item())
    return scores


def exact_match_accuracy(
    outputs: dict[str, "torch.Tensor"],
    targets: dict[str, "torch.Tensor"],
) -> float:
    _ensure_torch()
    matches = []
    for field in outputs.keys():
        predictions = outputs[field].argmax(dim=1)
        matches.append(predictions == targets[field])
    stacked = torch.stack(matches, dim=0)
    return float(stacked.all(dim=0).float().mean().item())
