from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from auto_app.config import UPLOADS_DIR


ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


async def save_uploaded_image(file: UploadFile) -> dict:
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image type. Use JPEG, PNG, or WEBP.",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ALLOWED_IMAGE_CONTENT_TYPES[content_type]
    filename = f"{uuid4().hex}{suffix}"
    destination = UPLOADS_DIR / filename
    destination.write_bytes(payload)

    return {
        "filename": filename,
        "content_type": content_type,
        "saved_path": str(destination),
        "file_size_bytes": len(payload),
    }


