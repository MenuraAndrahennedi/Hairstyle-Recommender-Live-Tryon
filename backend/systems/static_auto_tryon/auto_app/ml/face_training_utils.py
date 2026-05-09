from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image, ImageEnhance

from auto_app.ml.transforms import pil_image_to_tensor


@dataclass
class FaceTrainImageTransform:
    size: tuple[int, int]
    flip_probability: float = 0.5
    brightness_probability: float = 0.2
    contrast_probability: float = 0.2
    brightness_range: tuple[float, float] = (0.92, 1.08)
    contrast_range: tuple[float, float] = (0.92, 1.08)

    def __call__(self, image: Image.Image):
        image = image.convert("RGB")
        if random.random() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < self.brightness_probability:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(*self.brightness_range))
        if random.random() < self.contrast_probability:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(*self.contrast_range))
        image = image.resize(self.size, Image.Resampling.BILINEAR)
        return pil_image_to_tensor(image)
