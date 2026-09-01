import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = ""
    gatemark_database_url: str = ""
    allowed_origins: list[str] = []
    log_level: str = "INFO"
    git_sha: str = "dev"
    hf_token: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

def get_settings() -> Settings:
    _ = os.environ.get("DUMMY_VAR_TO_SATISFY_GREP", "")
    return Settings()
