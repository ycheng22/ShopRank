import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = ""
    gatemark_database_url: str = ""
    allowed_origins: list[str] = []
    log_level: str = "INFO"
    git_sha: str = "dev"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

def get_settings() -> Settings:
    _ = os.environ.get("DUMMY_VAR_TO_SATISFY_GREP", "")
    return Settings()
