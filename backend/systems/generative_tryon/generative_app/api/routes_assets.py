from __future__ import annotations

from fastapi import APIRouter

from generative_app.bootstrap_static import ensure_static_2d_on_path
from generative_app.config import media_url_for_path
from generative_app.models.schemas import AssetBankSummaryResponse, AssetItem, AssetListResponse


ensure_static_2d_on_path()

from support_app.core.asset_bank import asset_bank_summary, load_asset_bank  # noqa: E402


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
                gender_suitability=asset.gender_suitability,
                image_path=asset.image_path,
                mask_path=asset.mask_path,
                image_url=media_url_for_path(asset.image_path),
                mask_url=media_url_for_path(asset.mask_path),
                normalized_attributes=asset.normalized_attributes,
            )
            for asset in load_asset_bank()
        ]
    )
