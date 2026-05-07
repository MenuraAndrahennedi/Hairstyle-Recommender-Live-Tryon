from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None

try:
    from torchvision import models as tv_models
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    tv_models = None


def _ensure_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(
            "PyTorch is required for hairstyle attribute models. "
            "Install the packages from the project root requirements.txt before running the notebooks."
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


    class SmallConvBackbone(nn.Module):
        def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
            super().__init__()
            self.stem = ConvStem(in_channels, base_channels)
            self.stage2 = ConvStage(base_channels, base_channels * 2)
            self.stage3 = ConvStage(base_channels * 2, base_channels * 4)
            self.stage4 = ConvStage(base_channels * 4, base_channels * 8)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.out_features = base_channels * 8

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            features = self.stem(inputs)
            features = self.stage2(features)
            features = self.stage3(features)
            features = self.stage4(features)
            return self.pool(features).flatten(1)


    class TorchvisionBackbone(nn.Module):
        def __init__(self, architecture: str, in_channels: int = 3) -> None:
            super().__init__()
            if tv_models is None:
                raise ModuleNotFoundError(
                    "torchvision is required for ResNet-based attribute models. "
                    "Install the packages from requirements.txt before running training."
                )

            constructors = {
                "resnet18": tv_models.resnet18,
                "resnet34": tv_models.resnet34,
            }
            if architecture not in constructors:
                raise ValueError(f"Unsupported backbone architecture: {architecture}")

            backbone = constructors[architecture](weights=None)
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(
                    in_channels,
                    backbone.conv1.out_channels,
                    kernel_size=backbone.conv1.kernel_size,
                    stride=backbone.conv1.stride,
                    padding=backbone.conv1.padding,
                    bias=False,
                )
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.out_features = int(backbone.fc.in_features)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            return self.encoder(inputs).flatten(1)


    class HairstyleAttributeModel(nn.Module):
        def __init__(
            self,
            attribute_vocab_sizes: dict[str, int],
            in_channels: int = 3,
            base_channels: int = 32,
            dropout: float = 0.2,
            architecture: str = "cnn_small",
        ) -> None:
            super().__init__()
            self.attribute_vocab_sizes = dict(attribute_vocab_sizes)
            self.architecture = architecture
            self.in_channels = in_channels
            self.base_channels = base_channels
            self.dropout_rate = dropout

            if architecture == "cnn_small":
                self.backbone = SmallConvBackbone(in_channels=in_channels, base_channels=base_channels)
            else:
                self.backbone = TorchvisionBackbone(architecture=architecture, in_channels=in_channels)

            self.dropout = nn.Dropout(dropout)
            self.heads = nn.ModuleDict(
                {
                    field: nn.Linear(self.backbone.out_features, size)
                    for field, size in self.attribute_vocab_sizes.items()
                }
            )

        def forward(self, inputs: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            pooled = self.dropout(self.backbone(inputs))
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
    architecture: str = "cnn_small",
) -> HairstyleAttributeModel:
    _ensure_torch()
    if not attribute_vocab_sizes:
        raise ValueError("attribute_vocab_sizes cannot be empty.")
    return HairstyleAttributeModel(
        attribute_vocab_sizes=attribute_vocab_sizes,
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
        architecture=architecture,
    )


def multitask_cross_entropy(
    outputs: dict[str, "torch.Tensor"],
    targets: dict[str, "torch.Tensor"],
    class_weights: dict[str, "torch.Tensor"] | None = None,
    label_smoothing: float = 0.0,
) -> "torch.Tensor":
    _ensure_torch()
    losses = []
    for field in outputs.keys():
        weights = None if class_weights is None else class_weights.get(field)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        losses.append(criterion(outputs[field], targets[field]))
    return sum(losses)
