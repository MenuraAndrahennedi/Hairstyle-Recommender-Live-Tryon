from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from support_app.config import TRYON_DIR
from support_app.core.tryon_2d_engine import (
    _apply_asset_mask,
    _average_point,
    _build_subject_hair_region_mask,
    _cleanup_static_tryon_asset,
    _crop_asset_to_mask_bounds,
    _estimate_anchor_data,
    _estimate_background_rgb,
    _landmark_by_index,
    _pixel_point,
    _resolve_tryon_asset_paths,
)
from support_app.models.schemas import AssetMetadata, FaceAnalysisResult


GENERATIVE_PROTOTYPE_DIR = TRYON_DIR / "generative_prototype"


def _subject_anchor_payload(face_analysis: FaceAnalysisResult) -> dict[str, object]:
    image_width = face_analysis.image_width
    image_height = face_analysis.image_height
    landmarks = face_analysis.landmarks
    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    anchor = _estimate_anchor_data(face_analysis)

    left_temple = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in [127, 234]]
    )
    right_temple = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in [356, 454]]
    )
    left_eye = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in [33, 133]]
    )
    right_eye = _average_point(
        [_pixel_point(_landmark_by_index(landmarks, idx), image_width, image_height) for idx in [362, 263]]
    )
    forehead = _pixel_point(_landmark_by_index(landmarks, 10), image_width, image_height)
    chin = _pixel_point(_landmark_by_index(landmarks, 152), image_width, image_height)

    scalp_band_y = forehead[1] - face_bbox["height"] * 0.10
    scalp_left = (anchor["temple_center_x"] - anchor["temple_span"] * 0.72, scalp_band_y)
    scalp_right = (anchor["temple_center_x"] + anchor["temple_span"] * 0.72, scalp_band_y)

    return {
        "face_bbox": face_bbox,
        "left_temple": [float(left_temple[0]), float(left_temple[1])],
        "right_temple": [float(right_temple[0]), float(right_temple[1])],
        "left_eye": [float(left_eye[0]), float(left_eye[1])],
        "right_eye": [float(right_eye[0]), float(right_eye[1])],
        "forehead": [float(forehead[0]), float(forehead[1])],
        "chin": [float(chin[0]), float(chin[1])],
        "scalp_band_left": [float(scalp_left[0]), float(scalp_left[1])],
        "scalp_band_right": [float(scalp_right[0]), float(scalp_right[1])],
        "temple_span": float(anchor["temple_span"]),
        "eye_distance": float(anchor["eye_distance"]),
        "roll_degrees": float(anchor["roll_degrees"]),
    }


def _build_generative_inpaint_mask(
    canvas_size: tuple[int, int],
    face_analysis: FaceAnalysisResult,
    subject_hair_mask: Image.Image | None,
) -> Image.Image | None:
    subject_region = _build_subject_hair_region_mask(canvas_size, subject_hair_mask)
    if subject_region is None:
        return None

    face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}
    expanded = subject_region.filter(ImageFilter.MaxFilter(size=15))
    expanded_arr = np.asarray(expanded, dtype=np.uint8)

    overlay = Image.new("L", canvas_size, 0)
    draw = ImageDraw.Draw(overlay)
    left = max(int(face_bbox["x"] - face_bbox["width"] * 0.18), 0)
    top = max(int(face_bbox["y"] - face_bbox["height"] * 0.55), 0)
    right = min(int(face_bbox["x"] + face_bbox["width"] * 1.18), canvas_size[0])
    bottom = min(int(face_bbox["y"] + face_bbox["height"] * 0.30), canvas_size[1])
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(int(face_bbox["width"] * 0.18), 10),
        fill=255,
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=6))
    combined = np.maximum(expanded_arr, np.asarray(overlay, dtype=np.uint8))
    return Image.fromarray(combined, mode="L")


def _erase_subject_region(
    input_image: Image.Image,
    inpaint_mask: Image.Image | None,
) -> Image.Image:
    if inpaint_mask is None:
        return input_image.convert("RGBA")
    background_rgb = _estimate_background_rgb(input_image, inpaint_mask)
    base = input_image.convert("RGBA").copy()
    fill = Image.new("RGBA", input_image.size, (*background_rgb, 255))
    soft_mask = inpaint_mask.filter(ImageFilter.GaussianBlur(radius=2.2))
    return Image.composite(fill, base, soft_mask)


def _prepare_reference_asset(asset: AssetMetadata) -> tuple[Image.Image, Image.Image]:
    asset_image_path, asset_mask_path = _resolve_tryon_asset_paths(
        asset,
        prefer_clean_variant=True,
    )
    ref_image = Image.open(asset_image_path).convert("RGBA")
    ref_mask = Image.open(asset_mask_path).convert("L")
    ref_image, ref_mask = _crop_asset_to_mask_bounds(ref_image, ref_mask, padding_ratio=0.12)
    ref_image, ref_mask = _apply_asset_mask(ref_image, ref_mask)
    ref_image, ref_mask = _cleanup_static_tryon_asset(ref_image, ref_mask, asset)
    return ref_image, ref_mask


def _build_preview(
    input_image: Image.Image,
    erased_subject: Image.Image,
    inpaint_mask: Image.Image | None,
    reference_image: Image.Image,
) -> Image.Image:
    subject_rgb = input_image.convert("RGB")
    erased_rgb = erased_subject.convert("RGB")
    mask_rgb = (
        Image.merge("RGB", (inpaint_mask, inpaint_mask, inpaint_mask))
        if inpaint_mask is not None
        else Image.new("RGB", input_image.size, (0, 0, 0))
    )
    ref_rgb = reference_image.convert("RGB")

    tile_size = input_image.size
    preview = Image.new("RGB", (tile_size[0] * 2, tile_size[1] * 2), (245, 245, 245))
    preview.paste(subject_rgb, (0, 0))
    preview.paste(mask_rgb.resize(tile_size, Image.Resampling.NEAREST), (tile_size[0], 0))
    preview.paste(erased_rgb, (0, tile_size[1]))
    preview.paste(ref_rgb.resize(tile_size, Image.Resampling.LANCZOS), (tile_size[0], tile_size[1]))
    return preview


def prepare_generative_tryon_prototype(
    input_image_path: str | Path | Image.Image,
    face_analysis: FaceAnalysisResult,
    asset: AssetMetadata,
    subject_hair_mask: Image.Image | None = None,
) -> dict[str, object] | None:
    if not face_analysis.face_detected or not face_analysis.face_bbox or not face_analysis.landmarks:
        return None

    if isinstance(input_image_path, Image.Image):
        input_image = input_image_path.convert("RGBA")
        input_stem = "generative_frame"
    else:
        input_path = Path(input_image_path)
        input_image = Image.open(input_path).convert("RGBA")
        input_stem = input_path.stem

    anchor_payload = _subject_anchor_payload(face_analysis)
    inpaint_mask = _build_generative_inpaint_mask(input_image.size, face_analysis, subject_hair_mask)
    erased_subject = _erase_subject_region(input_image, inpaint_mask)
    reference_image, reference_mask = _prepare_reference_asset(asset)
    preview = _build_preview(input_image, erased_subject, inpaint_mask, reference_image)

    run_dir = GENERATIVE_PROTOTYPE_DIR / f"{input_stem}_{asset.asset_id}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_out = run_dir / "subject_input.png"
    input_image.convert("RGB").save(input_out)

    erased_out = run_dir / "subject_erased.png"
    erased_subject.convert("RGB").save(erased_out)

    mask_out = run_dir / "inpaint_mask.png"
    if inpaint_mask is not None:
        inpaint_mask.save(mask_out)

    reference_out = run_dir / "reference_asset.png"
    reference_image.save(reference_out)
    reference_mask_out = run_dir / "reference_mask.png"
    reference_mask.save(reference_mask_out)

    preview_out = run_dir / "prototype_preview.png"
    preview.save(preview_out)

    manifest = {
        "prototype_type": "generative_tryon_scaffold",
        "subject": {
            "input_image_path": str(input_out),
            "erased_subject_path": str(erased_out),
            "inpaint_mask_path": str(mask_out) if inpaint_mask is not None else None,
            "anchors": anchor_payload,
        },
        "reference": {
            "asset_id": asset.asset_id,
            "image_path": str(reference_out),
            "mask_path": str(reference_mask_out),
            "normalized_attributes": asset.normalized_attributes.model_dump(),
        },
        "generation_hints": {
            "task": "Replace subject hairstyle using the reference hairstyle while preserving face identity and pose.",
            "preserve": [
                "face identity",
                "skin tone",
                "camera pose",
                "background"
            ],
            "edit_region": "hair and hairline only",
            "notes": [
                "Use inpaint_mask.png as the editable region.",
                "Use the scalp band anchors as the target placement guide.",
                "Reference hairstyle should influence silhouette, curl, bangs, and volume."
            ],
        },
    }
    manifest_out = run_dir / "prototype_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "asset_id": asset.asset_id,
        "input_image_path": str(input_out),
        "erased_subject_path": str(erased_out),
        "inpaint_mask_path": str(mask_out) if inpaint_mask is not None else None,
        "reference_image_path": str(reference_out),
        "reference_mask_path": str(reference_mask_out),
        "preview_path": str(preview_out),
        "manifest_path": str(manifest_out),
        "face_detected": True,
        "face_bbox": face_analysis.face_bbox,
    }

