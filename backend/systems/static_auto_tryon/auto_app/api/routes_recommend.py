from __future__ import annotations

from fastapi import APIRouter

from auto_app.core.recommender import recommend_hairstyles
from auto_app.models.schemas import RecommendationRequest, RecommendationResponse


router = APIRouter(prefix="/api", tags=["recommend"])


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    return recommend_hairstyles(
        face_attributes=request.face_attributes,
        preferences=request.preferences,
        top_k=request.top_k,
    )

