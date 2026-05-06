from __future__ import annotations

from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None

from app.ml.hairstyle_attribute_model import build_attribute_model
from app.ml.hair_segmentation_model import build_segmentation_model


def _ensure_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for ML inference helpers. "
            "Install the packages from backend/systems/static_2d/requirements.txt before running the notebooks."
        )


def segmentation_logits_to_mask(logits: "torch.Tensor", threshold: float = 0.5) -> "torch.Tensor":
    _ensure_torch()
    return (torch.sigmoid(logits) >= threshold).float()


def load_segmentation_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    model_name: str = "unet",
    base_channels: int = 32,
) -> object:
    _ensure_torch()
    model = build_segmentation_model(model_name=model_name, base_channels=base_channels)
    payload = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_hairstyle_attribute_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[object, dict[str, dict[int, str]], tuple[str, ...], dict[str, object]]:
    _ensure_torch()
    payload = torch.load(Path(checkpoint_path), map_location=device)
    label_vocab = payload["label_vocab"]
    core_fields = tuple(payload.get("core_fields", tuple(label_vocab.keys())))
    attribute_vocab_sizes = {field: len(label_vocab[field]) for field in core_fields}
    model = build_attribute_model(attribute_vocab_sizes=attribute_vocab_sizes, base_channels=32, dropout=0.2)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    inverse_vocab = {
        field: {index: label for label, index in field_vocab.items()}
        for field, field_vocab in label_vocab.items()
    }
    return model, inverse_vocab, core_fields, payload


@torch.inference_mode() if torch is not None else (lambda func: func)
def predict_hair_mask(model: object, image_tensor: "torch.Tensor", threshold: float = 0.5) -> "torch.Tensor":
    _ensure_torch()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    logits = model(image_tensor)
    return segmentation_logits_to_mask(logits, threshold=threshold)


@torch.inference_mode() if torch is not None else (lambda func: func)
def predict_hairstyle_attributes(
    model: object,
    image_tensor: "torch.Tensor",
    label_vocab: dict[str, dict[int, str]],
) -> dict[str, dict[str, float | str]]:
    _ensure_torch()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    outputs = model(image_tensor)
    predictions: dict[str, dict[str, float | str]] = {}
    for field, logits in outputs.items():
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)
        label = label_vocab[field][int(index.item())]
        predictions[field] = {
            "label": label,
            "confidence": float(confidence.item()),
        }
    return predictions
