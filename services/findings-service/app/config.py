from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    findings_db_url: str = "sqlite+aiosqlite:///:memory:"
    # No insecure default — set SESSION_API_KEY (or PENTEST_ALLOW_DEV_SESSION_KEY=1
    # only when intentionally using the scaffold key in local/dev).
    session_api_key: str = ""
    defectdojo_api_url: str = "https://defectdojo.heimdall.local"
    defectdojo_api_token: str = ""
    default_pentest_profile: str = "pentester"


@lru_cache
def get_settings() -> Settings:
    return Settings()
