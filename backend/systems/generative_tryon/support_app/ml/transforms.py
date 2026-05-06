from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None


def _ensure_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for tensor-based transforms. "
            "Install the packages from backend/systems/static_2d/requirements.txt before running the notebooks."
        )


def pil_image_to_tensor(image: Image.Image) -> "torch.Tensor":
    _ensure_torch()
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def pil_mask_to_tensor(mask: Image.Image) -> "torch.Tensor":
    _ensure_torch()
    array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0).contiguous()


@dataclass
class ResizeImageAndMask:
    size: tuple[int, int]

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple["torch.Tensor", "torch.Tensor"]:
        resized_image = image.resize(self.size, Image.Resampling.BILINEAR)
        resized_mask = mask.resize(self.size, Image.Resampling.NEAREST)
        return pil_image_to_tensor(resized_image), pil_mask_to_tensor(resized_mask)


@dataclass
class ResizeImage:
    size: tuple[int, int]

    def __call__(self, image: Image.Image) -> "torch.Tensor":
        resized = image.resize(self.size, Image.Resampling.BILINEAR)
        return pil_image_to_tensor(resized)


@dataclass
class RandomHorizontalFlipPair:
    probability: float = 0.5

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < self.probability:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image, mask


class ComposeImageMaskTransforms:
    def __init__(self, transforms: list[object]) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[object, object]:
        current_image: object = image
        current_mask: object = mask
        for transform in self.transforms:
            current_image, current_mask = transform(current_image, current_mask)
        return current_image, current_mask


class ComposeImageTransforms:
    def __init__(self, transforms: list[object]) -> None:
        self.transforms = transforms

    def __call__(self, image: Image.Image) -> object:
        current: object = image
        for transform in self.transforms:
            current = transform(current)
        return current
