from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None

def _ensure_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for hairstyle attribute models. "
            "Install the packages from backend/systems/static_2d/requirements.txt before running the notebooks."
        )


if nn is not None:
    class ConvStem(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            return self.layers(inputs)


    class ConvStage(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            return self.layers(inputs)


    class HairstyleAttributeModel(nn.Module):
        def __init__(
            self,
            attribute_vocab_sizes: dict[str, int],
            in_channels: int = 3,
            base_channels: int = 32,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.attribute_vocab_sizes = attribute_vocab_sizes
            self.stem = ConvStem(in_channels, base_channels)
            self.stage2 = ConvStage(base_channels, base_channels * 2)
            self.stage3 = ConvStage(base_channels * 2, base_channels * 4)
            self.stage4 = ConvStage(base_channels * 4, base_channels * 8)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleDict(
                {
                    field: nn.Linear(base_channels * 8, size)
                    for field, size in attribute_vocab_sizes.items()
                }
            )

        def forward(self, inputs: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            features = self.stem(inputs)
            features = self.stage2(features)
            features = self.stage3(features)
            features = self.stage4(features)
            pooled = self.pool(features).flatten(1)
            pooled = self.dropout(pooled)
            return {field: head(pooled) for field, head in self.heads.items()}
else:
    class HairstyleAttributeModel:  # pragma: no cover - runtime fallback
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _ensure_torch()


def build_attribute_model(
    attribute_vocab_sizes: dict[str, int],
    in_channels: int = 3,
    base_channels: int = 32,
    dropout: float = 0.2,
) -> HairstyleAttributeModel:
    _ensure_torch()
    if not attribute_vocab_sizes:
        raise ValueError("attribute_vocab_sizes cannot be empty.")
    return HairstyleAttributeModel(
        attribute_vocab_sizes=attribute_vocab_sizes,
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
    )


def multitask_cross_entropy(
    outputs: dict[str, "torch.Tensor"],
    targets: dict[str, "torch.Tensor"],
) -> "torch.Tensor":
    _ensure_torch()
    criterion = nn.CrossEntropyLoss()
    losses = [criterion(outputs[field], targets[field]) for field in outputs.keys()]
    return sum(losses)
