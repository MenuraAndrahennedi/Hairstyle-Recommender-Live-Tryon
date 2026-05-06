from __future__ import annotations

from manual_app.models.schemas import FaceLandmark


def _band_width(
    landmarks: list[FaceLandmark],
    image_width: int,
    y_min: float,
    y_max: float,
    lower_ratio: float,
    upper_ratio: float,
) -> float:
    band_start = y_min + (y_max - y_min) * lower_ratio
    band_end = y_min + (y_max - y_min) * upper_ratio
    xs = [
        point.x * image_width
        for point in landmarks
        if band_start <= (point.y * 1.0) <= band_end
    ]
    if len(xs) < 2:
        return 0.0
    return max(xs) - min(xs)


def _bucket(value: float, thresholds: tuple[float, float], labels: tuple[str, str, str]) -> str:
    low, high = thresholds
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def _classify_face_shape(
    aspect_ratio: float,
    forehead_to_jaw_ratio: float,
    cheekbone_to_jaw_ratio: float,
    chin_to_jaw_ratio: float,
) -> str:
    if aspect_ratio >= 1.42:
        return "long"
    if forehead_to_jaw_ratio >= 1.12 and cheekbone_to_jaw_ratio >= 1.05:
        return "heart"
    if 0.94 <= forehead_to_jaw_ratio <= 1.06 and cheekbone_to_jaw_ratio <= 1.06 and chin_to_jaw_ratio <= 0.72:
        return "square"
    if aspect_ratio <= 1.18 and cheekbone_to_jaw_ratio >= 1.08:
        return "round"
    if 1.18 < aspect_ratio < 1.42 and 0.88 <= forehead_to_jaw_ratio <= 1.12:
        return "oval"
    return "unknown"


def compute_face_geometry(
    landmarks: list[FaceLandmark],
    image_width: int,
    image_height: int,
) -> tuple[dict[str, float | str], dict[str, str]] | tuple[None, None]:
    if not landmarks:
        return None, None

    xs = [point.x * image_width for point in landmarks]
    ys = [point.y for point in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    face_width = x_max - x_min
    face_height = (y_max - y_min) * image_height
    if face_width <= 0 or face_height <= 0:
        return None, None

    forehead_width = _band_width(landmarks, image_width, y_min, y_max, 0.18, 0.30)
    cheekbone_width = _band_width(landmarks, image_width, y_min, y_max, 0.38, 0.52)
    jaw_width = _band_width(landmarks, image_width, y_min, y_max, 0.68, 0.82)
    chin_width = _band_width(landmarks, image_width, y_min, y_max, 0.84, 0.95)

    aspect_ratio = face_height / face_width
    forehead_to_jaw_ratio = forehead_width / jaw_width if jaw_width else 0.0
    cheekbone_to_jaw_ratio = cheekbone_width / jaw_width if jaw_width else 0.0
    chin_to_jaw_ratio = chin_width / jaw_width if jaw_width else 0.0

    geometry = {
        "face_width_px": round(face_width, 2),
        "face_height_px": round(face_height, 2),
        "forehead_width_px": round(forehead_width, 2),
        "cheekbone_width_px": round(cheekbone_width, 2),
        "jaw_width_px": round(jaw_width, 2),
        "chin_width_px": round(chin_width, 2),
        "aspect_ratio": round(aspect_ratio, 4),
        "forehead_to_jaw_ratio": round(forehead_to_jaw_ratio, 4),
        "cheekbone_to_jaw_ratio": round(cheekbone_to_jaw_ratio, 4),
        "chin_to_jaw_ratio": round(chin_to_jaw_ratio, 4),
    }

    face_shape = _classify_face_shape(
        aspect_ratio=aspect_ratio,
        forehead_to_jaw_ratio=forehead_to_jaw_ratio,
        cheekbone_to_jaw_ratio=cheekbone_to_jaw_ratio,
        chin_to_jaw_ratio=chin_to_jaw_ratio,
    )
    face_attributes = {
        "face_shape": face_shape,
        "forehead": _bucket(forehead_to_jaw_ratio, (0.92, 1.08), ("small", "medium", "large")),
        "jaw": _bucket(chin_to_jaw_ratio, (0.68, 0.82), ("sharp", "medium", "soft")),
        "cheekbone": _bucket(cheekbone_to_jaw_ratio, (1.0, 1.1), ("low", "medium", "high")),
        "face_width": _bucket(face_width / image_width, (0.22, 0.30), ("narrow", "medium", "wide")),
    }

    return geometry, face_attributes

