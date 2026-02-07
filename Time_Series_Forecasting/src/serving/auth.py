"""API key authentication for FastAPI endpoints."""

import logging

from fastapi import Header, HTTPException

from src.config import settings

logger = logging.getLogger(__name__)


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """FastAPI dependency that validates the X-API-Key header.

    Skipped entirely when settings.api_key_enabled is False.
    """
    if not settings.api_key_enabled:
        return

    if x_api_key != settings.api_key_secret:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")
