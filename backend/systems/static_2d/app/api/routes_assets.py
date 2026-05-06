from __future__ import annotations

from fastapi import APIRouter

from app.config import media_url_for_path
from app.core.asset_bank import asset_bank_summary, load_asset_bank, load_live_top_tier_asset_bank
from app.core.recommender import infer_gender_suitability
from app.models.schemas import AssetBankSummaryResponse, AssetItem, AssetListResponse


router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("/summary", response_model=AssetBankSummaryResponse)
def get_asset_summary() -> AssetBankSummaryResponse:
    return AssetBankSummaryResponse(**asset_bank_summary())


@router.get("", response_model=AssetListResponse)
def list_assets() -> AssetListResponse:
    return AssetListResponse(
        assets=[
            AssetItem(
                asset_id=asset.asset_id,
                gender_suitability=infer_gender_suitability(asset),
                image_path=asset.image_path,
                mask_path=asset.mask_path,
                image_url=media_url_for_path(asset.image_path),
                mask_url=media_url_for_path(asset.mask_path),
                normalized_attributes=asset.normalized_attributes,
            )
            for asset in load_asset_bank()
        ]
    )


@router.get("/live-top-tier", response_model=AssetListResponse)
def list_live_top_tier_assets() -> AssetListResponse:
    return AssetListResponse(
        assets=[
            AssetItem(
                asset_id=asset.asset_id,
                gender_suitability=infer_gender_suitability(asset),
                image_path=asset.image_path,
                mask_path=asset.mask_path,
                image_url=media_url_for_path(asset.image_path),
                mask_url=media_url_for_path(asset.mask_path),
                normalized_attributes=asset.normalized_attributes,
            )
            for asset in load_live_top_tier_asset_bank()
        ]
    )
