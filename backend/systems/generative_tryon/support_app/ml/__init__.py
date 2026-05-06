from __future__ import annotations

from support_app.ml.datasets import (
    ATTRIBUTE_FIELDS,
    build_attribute_records_from_reviewed_assets,
    build_attribute_records_from_khairstyle,
    build_attribute_records_from_stage1_review,
    build_label_vocab_for_fields,
    build_segmentation_records_from_celebamask,
    train_val_split,
    write_jsonl_manifest,
)
from support_app.ml.face_to_hair_mapper import (
    FACE_FIELDS as MAPPER_FACE_FIELDS,
    HAIR_FIELDS as MAPPER_HAIR_FIELDS,
    build_face_to_hair_training_records,
    build_mapper_payload,
    evaluate_mapper_predictions,
    load_mapper_payload,
    recommend_assets_for_face,
    save_mapper_payload,
    split_face_to_hair_records,
    write_mapper_records,
)
from support_app.ml.hair_segmentation_model import HairSegmentationUNet, build_segmentation_model
from support_app.ml.hairstyle_attribute_model import HairstyleAttributeModel, build_attribute_model
from support_app.ml.khairstyle_translation import (
    FIELD_TRANSLATIONS,
    build_normalized_attributes,
    repair_mojibake,
    translate_labels,
    translate_value,
)

__all__ = [
    "ATTRIBUTE_FIELDS",
    "FIELD_TRANSLATIONS",
    "HairSegmentationUNet",
    "HairstyleAttributeModel",
    "MAPPER_FACE_FIELDS",
    "MAPPER_HAIR_FIELDS",
    "build_attribute_model",
    "build_face_to_hair_training_records",
    "build_attribute_records_from_khairstyle",
    "build_attribute_records_from_reviewed_assets",
    "build_normalized_attributes",
    "build_mapper_payload",
    "build_attribute_records_from_stage1_review",
    "build_label_vocab_for_fields",
    "build_segmentation_model",
    "build_segmentation_records_from_celebamask",
    "evaluate_mapper_predictions",
    "load_mapper_payload",
    "recommend_assets_for_face",
    "repair_mojibake",
    "save_mapper_payload",
    "split_face_to_hair_records",
    "train_val_split",
    "translate_labels",
    "translate_value",
    "write_mapper_records",
    "write_jsonl_manifest",
]

