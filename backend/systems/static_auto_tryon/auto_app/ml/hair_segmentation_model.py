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
            "PyTorch is required for segmentation models. "
            "Install the packages from the project root requirements.txt before running the notebooks."
        )


if nn is not None:
    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            return self.layers(inputs)


    class UpBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = ConvBlock(out_channels + skip_channels, out_channels)

        def forward(self, inputs: "torch.Tensor", skip: "torch.Tensor") -> "torch.Tensor":
            upsampled = self.up(inputs)
            if upsampled.shape[-2:] != skip.shape[-2:]:
                upsampled = torch.nn.functional.interpolate(
                    upsampled,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return self.conv(torch.cat([upsampled, skip], dim=1))


    class HairSegmentationUNet(nn.Module):
        def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
            super().__init__()
            self.enc1 = ConvBlock(in_channels, base_channels)
            self.enc2 = ConvBlock(base_channels, base_channels * 2)
            self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
            self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
            self.up4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
            self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
            self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
            self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
            self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            skip1 = self.enc1(inputs)
            skip2 = self.enc2(self.pool(skip1))
            skip3 = self.enc3(self.pool(skip2))
            skip4 = self.enc4(self.pool(skip3))
            bottleneck = self.bottleneck(self.pool(skip4))
            decoded = self.up4(bottleneck, skip4)
            decoded = self.up3(decoded, skip3)
            decoded = self.up2(decoded, skip2)
            decoded = self.up1(decoded, skip1)
            return self.head(decoded)
else:
    class HairSegmentationUNet:  # pragma: no cover - runtime fallback
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _ensure_torch()


def build_segmentation_model(
    model_name: str = "unet",
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
) -> HairSegmentationUNet:
    _ensure_torch()
    normalized_name = model_name.strip().lower()
    if normalized_name != "unet":
        raise ValueError(f"Unsupported segmentation model '{model_name}'. Version 1 currently supports 'unet'.")
    return HairSegmentationUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )
