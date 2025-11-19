from __future__ import annotations

import os

from pydantic import BaseModel


class Settings(BaseModel):
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-change-me")
    ACCESS_TOKEN_EXPIRE_MIN: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MIN", "60"))
    ALLOWED_MIME: set[str] = {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/csv",
    }
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))

    CORS_ALLOW_ORIGINS: list[str] = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8501").split(
        ","
    )


settings = Settings()
