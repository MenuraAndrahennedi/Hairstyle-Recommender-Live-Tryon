from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AnalyzeFaceResponse(BaseModel):
    face_detected: bool
    message: str
    filename: str
    content_type: str
    saved_path: str
    file_size_bytes: int
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    landmark_count: Optional[int] = None
    landmarks_path: Optional[str] = None
    face_bbox: Optional[Dict[str, int]] = None
    geometry: Optional[Dict[str, float | str]] = None
    face_attributes: Optional[Dict[str, str]] = None


class FaceLandmark(BaseModel):
    index: int
    x: float
    y: float
    z: float


class FaceAnalysisResult(BaseModel):
    face_detected: bool
    message: str
    image_width: int
    image_height: int
    landmark_count: int
    landmarks_path: Optional[str]
    face_bbox: Optional[Dict[str, int]]
    landmarks: List[FaceLandmark]
    geometry: Optional[Dict[str, float | str]] = None
    face_attributes: Optional[Dict[str, str]] = None


class NormalizedAttributes(BaseModel):
    length: str
    curl: str
    bang: str
    volume: str
    side_hair: str
    color: str
    style_family: str


class AssetQuality(BaseModel):
    front: bool
    horizontal: int
    vertical: str


class AssetMetadata(BaseModel):
    asset_id: str
    source_dataset: str
    gender_suitability: Optional[str] = None
    raw_label_path: str
    raw_image_path: str
    image_path: str
    mask_path: str
    translated_labels: Dict[str, str]
    normalized_attributes: NormalizedAttributes
    quality: AssetQuality


class FaceAttributes(BaseModel):
    face_shape: str
    forehead: str
    jaw: str
    cheekbone: str
    face_width: str


class RecommendationPreferences(BaseModel):
    preferred_length: Optional[str] = None
    preferred_color: Optional[str] = None
    allow_bangs: bool = True
    preferred_style_family: Optional[str] = None
    target_gender: Optional[str] = "any"


class RecommendationRequest(BaseModel):
    face_attributes: FaceAttributes
    preferences: Optional[RecommendationPreferences] = None
    top_k: int = 5


class RecommendationItem(BaseModel):
    asset_id: str
    score: float
    reason: str
    gender_suitability: Optional[str] = None
    image_path: str
    mask_path: str
    image_url: Optional[str] = None
    mask_url: Optional[str] = None
    normalized_attributes: NormalizedAttributes


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]


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
    assets: List[AssetItem]


class TryOnResponse(BaseModel):
    success: bool
    message: str
    asset_id: str
    filename: str
    saved_input_path: str
    saved_input_url: Optional[str] = None
    output_image_path: Optional[str] = None
    output_image_url: Optional[str] = None
    segmentation_mask_path: Optional[str] = None
    segmentation_mask_url: Optional[str] = None
    segmentation_used: bool = False
    face_detected: bool
    face_bbox: Optional[Dict[str, int]] = None


class HairSegmentationPredictionResponse(BaseModel):
    success: bool
    message: str
    filename: str
    saved_input_path: str
    saved_input_url: Optional[str] = None
    output_mask_path: Optional[str] = None
    output_mask_url: Optional[str] = None
    checkpoint_path: str
    threshold: float


class GenerativeTryOnPrototypeResponse(BaseModel):
    success: bool
    message: str
    asset_id: str
    input_image_path: str
    input_image_url: Optional[str] = None
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
    face_detected: bool
    face_bbox: Optional[Dict[str, int]] = None
