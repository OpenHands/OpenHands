from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    findings_db_url: str = "sqlite+aiosqlite:///:memory:"
    session_api_key: str = "dev-session-key"
    defectdojo_api_url: str = "https://defectdojo.heimdall.local"
    defectdojo_api_token: str = ""
    default_pentest_profile: str = "pentester"


@lru_cache
def get_settings() -> Settings:
    return Settings()
