from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import BACKEND_ROOT


ALLOWED_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5174",
    "http://localhost:5174",
]


def build_subsystem_app(
    *,
    title: str,
    version: str,
    description: str,
) -> FastAPI:
    app = FastAPI(
        title=title,
        version=version,
        description=description,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/media", StaticFiles(directory=BACKEND_ROOT), name="media")
    return app
