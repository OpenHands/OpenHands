from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    findings_db_url: str = "sqlite+aiosqlite:///:memory:"
    # No insecure default — set SESSION_API_KEY (or PENTEST_ALLOW_DEV_SESSION_KEY=1
    # only when intentionally using the scaffold key in local/dev).
    session_api_key: str = ""
    # DefectDojo production Heimdall (mirror only — do not provision a new DD).
    defectdojo_api_url: str = "https://defectdojo.heimdall.local"
    defectdojo_api_token: str = ""
    defectdojo_product_type_default: str = "Pentest"
    defectdojo_verify_tls: bool = True
    defectdojo_timeout_seconds: float = 30.0
    defectdojo_max_retries: int = 3
    # Local/scaffold only — never set in production.
    defectdojo_dry_run: bool = False
    default_pentest_profile: str = "pentester"

    def defectdojo_configured(self) -> bool:
        return bool(self.defectdojo_api_token.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
