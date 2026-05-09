from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import FACE_LANDMARKER_PATH
from systems.static_auto_tryon.auto_app.config import (
    FULL_HAIR_ASSET_ROOT,
    REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL,
    media_url_for_path,
)
from systems.static_auto_tryon.auto_app.core.asset_bank import (
    asset_bank_summary,
    load_asset_bank,
)
from systems.static_auto_tryon.auto_app.core.face_geometry import compute_face_geometry
from systems.static_auto_tryon.auto_app.core.hair_segmentation import (
    predict_hair_mask_image_from_face_roi,
)
from systems.static_auto_tryon.auto_app.core.recommender import recommend_hairstyles
from systems.static_auto_tryon.auto_app.models.schemas import (
    AssetMetadata,
    FaceAnalysisResult,
    FaceAttributes,
    FaceLandmark,
    RecommendationPreferences,
)


CLASSIFIER_SIZE = 128
SEGMENTATION_SIZE = 64

HAIR_WIDTH_SCALE = 1.55
HAIR_Y_OFFSET = 0.55
ROTATION_STRENGTH = 0.4

SMOOTHING = 0.75
ANGLE_SMOOTHING = 0.80

MASK_BLUR = 45
PREDICTION_INTERVAL = 1.0
MASK_INTERVAL = 0.3

TRYON_CLEAN_ROOT = FULL_HAIR_ASSET_ROOT / "tryon_clean"
TRYON_CLEAN_IMAGE_DIR = TRYON_CLEAN_ROOT / "images"
TRYON_CLEAN_MASK_DIR = TRYON_CLEAN_ROOT / "masks"


def _lazy_imports() -> tuple[Any, Any, Any]:
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    import numpy as np  # type: ignore

    return cv2, mp, np


@dataclass
class ManualTuning:
    x_offset: int = 0
    y_offset: int = 0
    scale: float = 1.0
    rotation: int = 0


@lru_cache(maxsize=1)
def live_candidate_assets() -> list[AssetMetadata]:
    candidates: list[AssetMetadata] = []
    for asset in load_asset_bank():
        image_path, mask_path = tryon_clean_paths(asset)
        if image_path.exists() and mask_path.exists():
            candidates.append(asset)
    return candidates


def tryon_clean_paths(asset: AssetMetadata) -> tuple[Path, Path]:
    image_name = Path(asset.image_path).name
    mask_name = Path(asset.mask_path).name
    return TRYON_CLEAN_IMAGE_DIR / image_name, TRYON_CLEAN_MASK_DIR / mask_name


class Live2DDemoEngine:
    def __init__(self) -> None:
        self.cv2, self.mp, self.np = _lazy_imports()

        base_options = self.mp.tasks.BaseOptions
        face_landmarker = self.mp.tasks.vision.FaceLandmarker
        face_landmarker_options = self.mp.tasks.vision.FaceLandmarkerOptions
        running_mode = self.mp.tasks.vision.RunningMode

        options = face_landmarker_options(
            base_options=base_options(model_asset_path=str(FACE_LANDMARKER_PATH)),
            running_mode=running_mode.VIDEO,
            num_faces=1,
        )
        self.face_landmarker = face_landmarker.create_from_options(options)

        self.tuning = ManualTuning()
        self.smooth_x1: float | None = None
        self.smooth_y1: float | None = None
        self.smooth_angle: float | None = None

        self.current_face_analysis: FaceAnalysisResult | None = None
        self.current_recommendations: list[Any] = []
        self.selected_index = 0
        self.current_mask: Image.Image | None = None
        self.current_mask_array: Any = None

        self.last_prediction_time = 0.0
        self.last_mask_time = 0.0

    def close(self) -> None:
        self.face_landmarker.close()

    def smooth_value(self, previous: float | None, current: float, smoothing: float) -> float:
        if previous is None:
            return current
        return previous * smoothing + current * (1 - smoothing)

    def get_point(self, landmarks: list[FaceLandmark], index: int, width: int, height: int) -> tuple[int, int]:
        return int(landmarks[index].x * width), int(landmarks[index].y * height)

    def calculate_face_angle(self, landmarks: list[FaceLandmark], width: int, height: int) -> float:
        left_eye = self.get_point(landmarks, 33, width, height)
        right_eye = self.get_point(landmarks, 263, width, height)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return math.degrees(math.atan2(dy, dx))

    def rotate_rgba(self, hair_rgb: Any, hair_alpha: Any, angle: float) -> tuple[Any, Any]:
        height, width = hair_rgb.shape[:2]
        rgba = self.np.dstack([hair_rgb, (hair_alpha * 255).astype(self.np.uint8)])

        center = (width // 2, height // 2)
        matrix = self.cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        matrix[0, 2] += (new_width / 2) - center[0]
        matrix[1, 2] += (new_height / 2) - center[1]

        rotated = self.cv2.warpAffine(
            rgba,
            matrix,
            (new_width, new_height),
            flags=self.cv2.INTER_LINEAR,
            borderMode=self.cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return rotated[:, :, :3], rotated[:, :, 3] / 255.0

    def suppress_original_hair(self, frame: Any, mask: Any) -> Any:
        mask_3 = self.np.repeat(mask[:, :, self.np.newaxis], 3, axis=2)
        blurred = self.cv2.GaussianBlur(frame, (45, 45), 0)
        suppressed = (blurred * 0.20).astype(self.np.uint8)
        return (mask_3 * suppressed + (1 - mask_3) * frame).astype(self.np.uint8)

    def overlay_hair(self, frame: Any, hair_rgb: Any, hair_alpha: Any, x1: float, y1: float) -> Any:
        height, width = frame.shape[:2]
        x1 = int(x1)
        y1 = int(y1)

        hair_height, hair_width = hair_rgb.shape[:2]
        x2 = x1 + hair_width
        y2 = y1 + hair_height

        if x1 < 0:
            hair_rgb = hair_rgb[:, -x1:]
            hair_alpha = hair_alpha[:, -x1:]
            x1 = 0
        if y1 < 0:
            hair_rgb = hair_rgb[-y1:, :]
            hair_alpha = hair_alpha[-y1:, :]
            y1 = 0
        if x2 > width:
            cut = x2 - width
            hair_rgb = hair_rgb[:, :-cut]
            hair_alpha = hair_alpha[:, :-cut]
            x2 = width
        if y2 > height:
            cut = y2 - height
            hair_rgb = hair_rgb[:-cut, :]
            hair_alpha = hair_alpha[:-cut, :]
            y2 = height

        if hair_rgb.size == 0 or hair_alpha.size == 0:
            return frame

        roi = frame[y1:y2, x1:x2]
        if roi.shape[:2] != hair_alpha.shape[:2]:
            return frame

        alpha = self.cv2.GaussianBlur(hair_alpha, (15, 15), 0)
        alpha = self.np.clip(alpha, 0, 1)

        for channel in range(3):
            roi[:, :, channel] = (
                hair_alpha * hair_rgb[:, :, channel]
                + (1 - alpha) * roi[:, :, channel]
            )

        frame[y1:y2, x1:x2] = roi
        return frame

    def _build_face_bbox(self, landmarks: list[FaceLandmark], width: int, height: int) -> dict[str, int] | None:
        if not landmarks:
            return None
        xs = [max(0, min(width - 1, int(point.x * width))) for point in landmarks]
        ys = [max(0, min(height - 1, int(point.y * height))) for point in landmarks]
        return {
            "x": min(xs),
            "y": min(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }

    def _analyze_frame(self, frame: Any) -> FaceAnalysisResult:
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return FaceAnalysisResult(
                face_detected=False,
                message="No face detected in the current frame.",
                image_width=width,
                image_height=height,
                landmark_count=0,
                landmarks_path=None,
                face_bbox=None,
                landmarks=[],
                geometry=None,
                face_attributes=None,
            )

        first_face = result.face_landmarks[0]
        landmarks = [
            FaceLandmark(index=index, x=point.x, y=point.y, z=point.z)
            for index, point in enumerate(first_face)
        ]
        face_bbox = self._build_face_bbox(landmarks, width, height)
        geometry, face_attributes = compute_face_geometry(landmarks, width, height)

        return FaceAnalysisResult(
            face_detected=True,
            message="Face landmarks detected successfully.",
            image_width=width,
            image_height=height,
            landmark_count=len(landmarks),
            landmarks_path=None,
            face_bbox=face_bbox,
            landmarks=landmarks,
            geometry=geometry,
            face_attributes=face_attributes,
        )

    def _refresh_recommendations(self, face_analysis: FaceAnalysisResult) -> None:
        if not face_analysis.face_attributes:
            self.current_recommendations = []
            self.selected_index = 0
            return

        face_attributes = FaceAttributes.model_validate(face_analysis.face_attributes)
        response = recommend_hairstyles(
            face_attributes,
            preferences=RecommendationPreferences(target_gender="any", allow_bangs=True),
            top_k=3,
            candidate_assets=live_candidate_assets(),
        )
        self.current_recommendations = response.recommendations
        if self.selected_index >= len(self.current_recommendations):
            self.selected_index = 0

    def _selected_asset(self) -> AssetMetadata | None:
        if not self.current_recommendations:
            return None
        selected_asset_id = self.current_recommendations[self.selected_index].asset_id
        for asset in live_candidate_assets():
            if asset.asset_id == selected_asset_id:
                return asset
        return None

    def set_selected_asset_by_id(self, asset_id: str) -> bool:
        for index, item in enumerate(self.current_recommendations):
            if item.asset_id == asset_id:
                self.selected_index = index
                self.reset_smoothing()
                return True
        return False

    def current_response_payload(self) -> dict[str, Any]:
        selected_asset = self._selected_asset()
        recommendations = []
        for item in self.current_recommendations:
            matched_asset = next(
                (asset for asset in live_candidate_assets() if asset.asset_id == item.asset_id),
                None,
            )
            if matched_asset is None:
                continue
            recommendations.append(
                {
                    "asset_id": item.asset_id,
                    "score": float(item.score),
                    "reason": item.reason,
                    "gender_suitability": matched_asset.gender_suitability,
                    "image_path": matched_asset.image_path,
                    "mask_path": matched_asset.mask_path,
                    "image_url": media_url_for_path(matched_asset.image_path),
                    "mask_url": media_url_for_path(matched_asset.mask_path),
                    "normalized_attributes": matched_asset.normalized_attributes.model_dump(),
                }
            )

        return {
            "face_detected": bool(self.current_face_analysis and self.current_face_analysis.face_detected),
            "face_bbox": self.current_face_analysis.face_bbox if self.current_face_analysis else None,
            "image_width": self.current_face_analysis.image_width if self.current_face_analysis else None,
            "image_height": self.current_face_analysis.image_height if self.current_face_analysis else None,
            "selected_asset_id": selected_asset.asset_id if selected_asset is not None else None,
            "recommendations": recommendations,
        }

    def load_project_hair_asset(self, asset: AssetMetadata) -> tuple[Any | None, Any | None]:
        image_path, mask_path = tryon_clean_paths(asset)
        if not image_path.exists() or not mask_path.exists():
            return None, None

        image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        mask = self.cv2.imread(str(mask_path), self.cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            return None, None

        hair_rgb = image
        hair_alpha = mask.astype("float32") / 255.0
        return hair_rgb, hair_alpha

    def _update_subject_mask(self, pil_image: Image.Image, face_analysis: FaceAnalysisResult) -> None:
        self.current_mask = predict_hair_mask_image_from_face_roi(
            pil_image,
            face_analysis.face_bbox,
        )
        self.current_mask_array = self.np.asarray(self.current_mask.convert("L"), dtype="float32") / 255.0
        self.current_mask_array = self.cv2.GaussianBlur(self.current_mask_array, (MASK_BLUR, MASK_BLUR), 0)
        self.current_mask_array = self.np.clip(self.current_mask_array, 0, 1)

    def reset_smoothing(self) -> None:
        self.smooth_x1 = None
        self.smooth_y1 = None
        self.smooth_angle = None

    def process_frame(self, frame: Any) -> Any:
        frame = self.cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        face_analysis = self._analyze_frame(frame)
        self.current_face_analysis = face_analysis

        if not face_analysis.face_detected:
            self.current_mask = None
            self.current_mask_array = None
            self.current_recommendations = []
            self.reset_smoothing()
            self.draw_empty_hud(frame)
            return frame

        now = time.time()
        if not self.current_recommendations or now - self.last_prediction_time > PREDICTION_INTERVAL:
            self._refresh_recommendations(face_analysis)
            self.last_prediction_time = now

        if not self.current_recommendations:
            self.draw_empty_hud(frame)
            return frame

        pil_image = Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)).convert("RGB")
        if self.current_mask is None or now - self.last_mask_time > MASK_INTERVAL:
            self._update_subject_mask(pil_image, face_analysis)
            self.last_mask_time = now

        if self.current_mask_array is not None:
            frame = self.suppress_original_hair(frame, self.current_mask_array)

        asset = self._selected_asset()
        if asset is not None:
            hair_rgb, hair_alpha = self.load_project_hair_asset(asset)
            if hair_rgb is not None and hair_alpha is not None:
                landmarks = face_analysis.landmarks
                face_bbox = face_analysis.face_bbox or {"x": 0, "y": 0, "width": 1, "height": 1}

                x_min = face_bbox["x"]
                x_max = face_bbox["x"] + face_bbox["width"]
                face_width = face_bbox["width"]

                forehead_y = int((landmarks[10].y + landmarks[151].y) / 2 * height)
                center_x = int((x_min + x_max) / 2)

                target_width = int(face_width * HAIR_WIDTH_SCALE * self.tuning.scale)
                scale = target_width / max(hair_rgb.shape[1], 1)
                new_width = max(int(hair_rgb.shape[1] * scale), 1)
                new_height = max(int(hair_rgb.shape[0] * scale), 1)

                hair_rgb = self.cv2.resize(hair_rgb, (new_width, new_height))
                hair_alpha = self.cv2.resize(hair_alpha, (new_width, new_height))

                raw_angle = self.calculate_face_angle(landmarks, width, height)
                raw_angle = (raw_angle * ROTATION_STRENGTH) + self.tuning.rotation
                self.smooth_angle = self.smooth_value(self.smooth_angle, raw_angle, ANGLE_SMOOTHING)

                rotated_rgb, rotated_alpha = self.rotate_rgba(
                    hair_rgb,
                    hair_alpha,
                    -self.smooth_angle,
                )

                rotated_height, rotated_width = rotated_rgb.shape[:2]
                raw_x1 = center_x - rotated_width // 2 + self.tuning.x_offset
                raw_y1 = forehead_y - int(HAIR_Y_OFFSET * rotated_height) + self.tuning.y_offset

                self.smooth_x1 = self.smooth_value(self.smooth_x1, raw_x1, SMOOTHING)
                self.smooth_y1 = self.smooth_value(self.smooth_y1, raw_y1, SMOOTHING)

                frame = self.overlay_hair(
                    frame,
                    rotated_rgb,
                    rotated_alpha,
                    self.smooth_x1,
                    self.smooth_y1,
                )

        self.draw_hud(frame)
        return frame

    def draw_empty_hud(self, frame: Any) -> None:
        self.cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    def draw_hud(self, frame: Any) -> None:
        selected_asset = self._selected_asset()
        selected_item = None
        if self.current_recommendations:
            selected_item = self.current_recommendations[self.selected_index]

        label = selected_asset.asset_id if selected_asset is not None else "No asset"
        confidence = (selected_item.score * 100.0) if selected_item is not None else 0.0

        self.cv2.putText(
            frame,
            f"{label} ({confidence:.1f}%)",
            (20, 35),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        self.cv2.putText(
            frame,
            f"Top picks: {len(self.current_recommendations)} | Clean bank: {len(live_candidate_assets())}",
            (20, 65),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        self.cv2.putText(
            frame,
            (
                f"x:{self.tuning.x_offset} y:{self.tuning.y_offset} "
                f"scale:{self.tuning.scale:.2f} rot:{self.tuning.rotation}"
            ),
            (20, 95),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        self.cv2.putText(
            frame,
            "Arrows/W/S/A/D tune | 1/2/3 pick | R cycle | T reset | Q quit",
            (20, frame.shape[0] - 20),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

    def handle_key(self, key: int) -> bool:
        # Windows OpenCV arrow key codes
        LEFT_KEYS = {81, 2424832, 65361}
        RIGHT_KEYS = {83, 2555904, 65363}
        UP_KEYS = {82, 2490368, 65362}
        DOWN_KEYS = {84, 2621440, 65364}

        # Arrow keys
        if key in LEFT_KEYS:
            self.tuning.x_offset -= 5
        elif key in RIGHT_KEYS:
            self.tuning.x_offset += 5
        elif key in UP_KEYS:
            self.tuning.y_offset -= 5
        elif key in DOWN_KEYS:
            self.tuning.y_offset += 5

        else:
            k = key & 0xFF

            if k == ord("q"):
                return False

            # Extra fallback movement keys
            elif k == ord("j"):
                self.tuning.x_offset -= 5
            elif k == ord("l"):
                self.tuning.x_offset += 5
            elif k == ord("i"):
                self.tuning.y_offset -= 5
            elif k == ord("k"):
                self.tuning.y_offset += 5

            elif k == ord("w"):
                self.tuning.scale += 0.05
            elif k == ord("s"):
                self.tuning.scale = max(0.3, self.tuning.scale - 0.05)

            elif k == ord("a"):
                self.tuning.rotation -= 3
            elif k == ord("d"):
                self.tuning.rotation += 3
            elif k == ord("r") and self.current_recommendations:
                self.selected_index = (self.selected_index + 1) % len(self.current_recommendations)
                self.reset_smoothing()

            elif k in (ord("1"), ord("2"), ord("3")) and self.current_recommendations:
                requested_index = int(chr(k)) - 1
                if requested_index < len(self.current_recommendations):
                    self.selected_index = requested_index
                    self.reset_smoothing()

            elif k == ord("t"):
                self.tuning = ManualTuning()
                self.reset_smoothing()

        return True

    def run_webcam(self) -> None:
        cap = None
        capture_attempts: list[tuple[int, str, Any]] = [(0, "default", 0)]

        if hasattr(self.cv2, "CAP_DSHOW"):
            for camera_index in range(4):
                capture_attempts.append((camera_index, "dshow", self.cv2.CAP_DSHOW))
        if hasattr(self.cv2, "CAP_MSMF"):
            for camera_index in range(4):
                capture_attempts.append((camera_index, "msmf", self.cv2.CAP_MSMF))
        for camera_index in range(1, 4):
            capture_attempts.append((camera_index, "default", camera_index))

        attempted_sources: list[str] = []
        for camera_index, backend_name, backend_flag in capture_attempts:
            candidate = (
                self.cv2.VideoCapture(camera_index)
                if backend_name == "default"
                else self.cv2.VideoCapture(camera_index, backend_flag)
            )
            if candidate.isOpened():
                cap = candidate
                print(f"Opened webcam with backend: {backend_name} at camera index: {camera_index}")
                break
            attempted_sources.append(f"{backend_name}:{camera_index}")
            candidate.release()

        if cap is None:
            raise RuntimeError(
                "Could not open webcam. Tried "
                + ", ".join(attempted_sources)
                + ". Close other apps using the camera and check Windows camera permissions."
            )

        print("\nControls:")
        print("Arrow keys = move hair")
        print("W / S = increase/decrease size")
        print("A / D = rotate left/right")
        print("1 / 2 / 3 = choose recommended asset")
        print("R = cycle recommendation")
        print("T = reset tuning")
        print("Q = quit")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = self.process_frame(frame)
                self.cv2.imshow("Live 2D Try-On (Original Overlay Logic)", frame)
                key = self.cv2.waitKeyEx(1)
                if key != -1:
                    # Optional: print key codes for debugging
                    # print("Pressed key:", key)

                    if not self.handle_key(key):
                        break
        finally:
            cap.release()
            self.cv2.destroyAllWindows()
            self.close()


def runtime_readiness() -> dict[str, Any]:
    summary = asset_bank_summary()
    required_paths = {
        "face_landmarker": FACE_LANDMARKER_PATH,
        "reviewed_render_safe_asset_bank": REVIEWED_RENDER_SAFE_ASSET_BANK_JSONL,
        "tryon_clean_image_dir": TRYON_CLEAN_IMAGE_DIR,
        "tryon_clean_mask_dir": TRYON_CLEAN_MASK_DIR,
    }
    missing = [name for name, path in required_paths.items() if not Path(path).exists()]
    candidates = live_candidate_assets()
    return {
        "ready": (not missing) and bool(candidates),
        "missing": missing,
        "asset_count": summary["asset_count"],
        "metadata_dir": summary["metadata_dir"],
        "tryon_clean_candidate_count": len(candidates),
        "tryon_clean_asset_root": str(TRYON_CLEAN_ROOT),
        "candidate_asset_ids": [asset.asset_id for asset in candidates],
    }
