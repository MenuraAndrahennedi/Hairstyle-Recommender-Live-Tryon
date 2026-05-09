from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from generative_app.bootstrap_static import ensure_static_2d_on_path


ensure_static_2d_on_path()

from support_app.models.schemas import NormalizedAttributes  # noqa: E402


class HealthResponse(BaseModel):
    status: str


class AssetBankSummaryResponse(BaseModel):
    asset_count: int
    metadata_dir: str


class AssetItem(BaseModel):
    asset_id: str
    gender_suitability: Optional[str] = None
    image_path: str
    mask_path: str
    image_url: str
    mask_url: str
    normalized_attributes: NormalizedAttributes


class AssetListResponse(BaseModel):
    assets: list[AssetItem]


class GenerativeTryOnPackageResponse(BaseModel):
    success: bool
    message: str
    asset_id: str
    input_image_path: str
    input_image_url: Optional[str] = None
    segmentation_mask_path: Optional[str] = None
    segmentation_mask_url: Optional[str] = None
    erased_subject_path: Optional[str] = None
    erased_subject_url: Optional[str] = None
    inpaint_mask_path: Optional[str] = None
    inpaint_mask_url: Optional[str] = None
    reference_image_path: Optional[str] = None
    reference_image_url: Optional[str] = None
    reference_mask_path: Optional[str] = None
    reference_mask_url: Optional[str] = None
    preview_path: Optional[str] = None
    preview_url: Optional[str] = None
    manifest_path: Optional[str] = None
    manifest_url: Optional[str] = None
    prompt_path: Optional[str] = None
    prompt_url: Optional[str] = None
    final_image_path: Optional[str] = None
    final_image_url: Optional[str] = None
    final_metadata_path: Optional[str] = None
    final_metadata_url: Optional[str] = None
    final_generation_completed: bool = False
    final_generation_error: Optional[str] = None
    face_detected: bool
    face_bbox: Optional[dict[str, int]] = None
