from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_FALLBACK_ANDROID_EMULATOR = "budtmo/docker-android:emulator_13.0"
_FALLBACK_MOBSF = "opensecurity/mobile-security-framework-mobsf:latest"


def _repo_root() -> Path:
    # app/config.py → engagement-manager → services → repo root
    return Path(__file__).resolve().parents[3]


def _defaults_images() -> dict[str, str]:
    path = _repo_root() / "config" / "defaults.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    images = data.get("images") or {}
    return {k: v for k, v in images.items() if isinstance(v, str)}


_IMAGES = _defaults_images()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    engmgr_db_url: str = "sqlite+aiosqlite:///:memory:"
    # No insecure default — set SESSION_API_KEY (or PENTEST_ALLOW_DEV_SESSION_KEY=1
    # only when intentionally using the scaffold key in local/dev).
    session_api_key: str = ""
    default_pentest_profile: str = "pentester"
    compose_work_dir: str = "/tmp/engmgr-compose"
    # When true, provisioner skips real docker compose (tests / scaffold)
    provisioner_dry_run: bool = True
    # Image pins — defaults.json is source of truth; env overrides for ops.
    android_emulator_image: str = _IMAGES.get(
        "androidEmulator", _FALLBACK_ANDROID_EMULATOR
    )
    mobsf_image: str = _IMAGES.get("mobsf", _FALLBACK_MOBSF)
    # MVP: without /dev/kvm, provision fails unless this flag is set.
    allow_slow_emulator: bool = False
    mcp_mobile_cmd: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
