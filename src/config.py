"""Environment and provider configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

_ENV_LOADED = False


def load_env() -> None:
    """Load `.env` from the project root once per process."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=True)
    except ImportError:
        if ENV_FILE.is_file():
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ[key] = value

    _ENV_LOADED = True


@lru_cache(maxsize=1)
def get_openai_api_key() -> str | None:
    load_env()
    for name in ("OPENAI_API_KEY", "openai_api_key"):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def get_openai_model() -> str:
    load_env()
    return os.getenv("OPENAI_MODEL", os.getenv("openai_model", "gpt-4o-mini")).strip()


def openai_configured() -> bool:
    return get_openai_api_key() is not None

@lru_cache(maxsize=1)
def get_gemini_api_key() -> str | None:
    load_env()
    for name in ("GEMINI_API_KEY", "gemini_api_key"):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def get_gemini_model() -> str:
    load_env()
    return os.getenv("GEMINI_MODEL", os.getenv("gemini_model", "gemini-3.6-flash")).strip()


def gemini_configured() -> bool:
    return get_gemini_api_key() is not None
