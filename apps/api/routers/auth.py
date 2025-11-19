from __future__ import annotations

import os
import time
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from apps.api.settings import settings

# simple single-user creds via env
USERNAME = os.getenv("ADMIN_USER", "admin")
PWD_HASH = os.getenv("ADMIN_PWD_HASH")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGO = "HS256"


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(sub: str, minutes: int) -> str:
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + minutes * 60}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGO)


def decode_token(tok: str) -> dict[str, Any]:
    return jwt.decode(tok, settings.SECRET_KEY, algorithms=[ALGO])


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = decode_token(token)
        return payload["sub"]
    except JWTError:
        raise HTTPException(401, "invalid or expired token")
