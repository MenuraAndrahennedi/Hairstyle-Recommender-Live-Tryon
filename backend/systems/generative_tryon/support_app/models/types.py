from __future__ import annotations

from typing import TypedDict


class ScoreBreakdown(TypedDict, total=False):
    learned_mapper_score: float
    face_compatibility_score: float
    user_preference_score: float
    hair_attribute_confidence_score: float
    asset_quality_score: float
