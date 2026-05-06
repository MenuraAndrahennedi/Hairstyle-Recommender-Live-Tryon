from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping


FIELD_TRANSLATIONS: dict[str, dict[str, str]] = {
    "basestyle": {
        "가르마": "parted",
        "기타남자스타일": "other_mens_style",
        "기타레이어드": "other_layered",
        "기타여자스타일": "other_womens_style",
        "남자일반숏": "men_general_short",
        "댄디": "dandy",
        "루프": "loop",
        "리젠트": "regent",
        "리프": "leaf",
        "미스티": "misty",
        "바디": "body_wave",
        "보브": "bob",
        "소프트투블럭댄디": "soft_two_block_dandy",
        "숏단발": "short_bob",
        "쉼표": "comma",
        "시스루댄디": "see_through_dandy",
        "여자일반숏": "women_general_short",
        "원랭스": "one_length",
        "포마드": "pomade",
        "히피": "hippie",
    },
    "basestyle-type": {
        "단": "short_type",
        "장": "long_type",
    },
    "length": {
        "남자": "men_style",
        "단발": "bob_length",
        "여숏": "women_short",
        "장발": "long_hair",
        "중발": "medium_hair",
    },
    "curl": {
        "C": "curl_code_c",
        "CC": "curl_code_cc",
        "CS": "curl_code_cs",
        "J": "curl_code_j",
        "S": "curl_code_s",
        "S3": "curl_code_s3",
        "SC": "curl_code_sc",
        "SS": "curl_code_ss",
        "X": "straight_code_x",
    },
    "bang": {
        "기타(남자 내림머리)": "other_mens_down_style",
        "살짝 넘긴스타일": "lightly_swept_style",
        "시스루": "see_through_bangs",
        "처피뱅": "choppy_bangs",
        "페이스라인에 통합": "integrated_into_face_line",
        "풀뱅": "full_bangs",
        "해당없음": "not_applicable",
    },
    "loss": {
        "부분탈모": "partial_hair_loss",
        "탈모": "hair_loss",
        "탈모아님": "no_hair_loss",
    },
    "side": {
        "원블럭": "one_block",
        "투블럭": "two_block",
        "해당없음": "not_applicable",
    },
    "color": {
        "기타": "other",
        "블랙": "black",
        "애쉬브라운": "ash_brown",
        "옴브레": "ombre",
        "자연갈색": "natural_brown",
        "적갈색": "reddish_brown",
        "황갈색": "yellowish_brown",
    },
    "partition": {
        "2:8": "part_2_8",
        "3:7": "part_3_7",
        "4:6": "part_4_6",
        "5:5": "part_5_5",
        "6:4": "part_6_4",
        "7:3": "part_7_3",
        "8:2": "part_8_2",
        "가르마없음": "no_part",
    },
    "sex": {
        "남": "male",
        "여": "female",
    },
    "before-after": {
        "after": "after",
        "before": "before",
    },
    "vertical": {
        "상": "upper",
        "중": "middle",
    },
    "hair-width": {
        "굵음": "thick",
        "보통": "medium",
        "얇음": "thin",
    },
    "natural-curl": {
        "강곱슬": "tightly_curly",
        "곱슬": "curly",
        "반곱슬": "semi_curly",
        "생머리": "straight",
    },
    "damage": {
        "건강모": "healthy",
        "극손상모": "severely_damaged",
        "버진": "virgin",
        "손상모": "damaged",
    },
}


def repair_mojibake(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired


def slugify_ascii(value: object) -> str:
    text = str(value).strip()
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    return slug or "unknown"


def translate_value(field: str, value: object) -> str:
    repaired = repair_mojibake(value)
    if repaired is None:
        return ""
    if not isinstance(repaired, str):
        return str(repaired)

    mapping = FIELD_TRANSLATIONS.get(field, {})
    if repaired in mapping:
        return mapping[repaired]

    if repaired.isascii():
        return slugify_ascii(repaired)
    return f"unmapped_{field}_{slugify_ascii(repaired)}"


def translate_labels(payload: Mapping[str, object]) -> dict[str, str]:
    translated: dict[str, str] = {}
    for field in FIELD_TRANSLATIONS:
        translated[field] = translate_value(field, payload.get(field))
    return translated


def build_normalized_attributes(translated_labels: Mapping[str, str]) -> dict[str, str]:
    base = translated_labels.get("basestyle", "")
    length = translated_labels.get("length", "")
    curl = translated_labels.get("curl", "")
    bang = translated_labels.get("bang", "")
    side = translated_labels.get("side", "")
    color = translated_labels.get("color", "")
    partition = translated_labels.get("partition", "")
    width = translated_labels.get("hair-width", "")

    normalized_length = "medium"
    if length in {"men_style", "women_short", "bob_length"}:
        normalized_length = "short"
    elif length == "long_hair":
        normalized_length = "long"

    normalized_curl = "wavy"
    if curl in {"straight_code_x"}:
        normalized_curl = "straight"
    elif curl in {"curl_code_j", "curl_code_s", "curl_code_ss"}:
        normalized_curl = "soft_wave"
    elif curl in {"curl_code_c", "curl_code_cc", "curl_code_cs", "curl_code_sc", "curl_code_s3"}:
        normalized_curl = "curly"

    normalized_bang = "none"
    if bang in {"see_through_bangs", "full_bangs", "choppy_bangs"}:
        normalized_bang = "fringe"
    elif bang in {"lightly_swept_style", "other_mens_down_style"}:
        normalized_bang = "side"

    normalized_volume = width if width in {"thin", "medium", "thick"} else "medium"
    normalized_side_hair = "covered" if side in {"one_block", "two_block"} else "open"
    normalized_color = color or "other"

    style_family = "custom"
    if base in {"parted", "comma", "regent", "pomade"}:
        style_family = "side_part"
    elif base in {"bob", "short_bob", "one_length"}:
        style_family = "bob"
    elif base in {"dandy", "soft_two_block_dandy", "see_through_dandy", "men_general_short"}:
        style_family = "dandy"
    elif base in {"leaf", "loop", "misty", "body_wave", "hippie"}:
        style_family = "wave"
    elif partition == "no_part":
        style_family = "no_part"

    return {
        "length": normalized_length,
        "curl": normalized_curl,
        "bang": normalized_bang,
        "volume": normalized_volume,
        "side_hair": normalized_side_hair,
        "color": normalized_color,
        "style_family": style_family,
    }


__all__ = [
    "FIELD_TRANSLATIONS",
    "build_normalized_attributes",
    "repair_mojibake",
    "translate_labels",
    "translate_value",
]
