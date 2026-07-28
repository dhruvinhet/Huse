"""Environment-backed application configuration."""

from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    """Validated configuration values used by the application."""

    model_config = ConfigDict(frozen=True)

    GEMINI_API_KEY: str
    OUTPUT_DIR: Path
    TEMP_DIR: Path
    LOG_LEVEL: str
    GEMINI_TIMEOUT_SECONDS: float | None
    FFMPEG_PATH: Path | None
    DEBUG_ARTIFACTS: bool
    DEBUG_DIR: Path
    PIPELINE_VERSION: str
    V2_MAX_REPAIR_ATTEMPTS: int
    V2_ENABLE_MULTIMODAL: bool


def _parse_optional_timeout(value: str | None) -> float | None:
    """Return seconds, or None when the API request deadline is disabled."""

    if value is None or value.strip().lower() in {"", "none", "unlimited", "0"}:
        return None
    timeout = float(value)
    if timeout < 0:
        raise ValueError("GEMINI_TIMEOUT_SECONDS cannot be negative")
    return timeout


settings = Settings(
    GEMINI_API_KEY=getenv("GEMINI_API_KEY", ""),
    OUTPUT_DIR=Path(getenv("OUTPUT_DIR", "outputs")),
    TEMP_DIR=Path(getenv("TEMP_DIR", "temp")),
    LOG_LEVEL=getenv("LOG_LEVEL", "INFO"),
    GEMINI_TIMEOUT_SECONDS=_parse_optional_timeout(
        getenv("GEMINI_TIMEOUT_SECONDS")
    ),
    FFMPEG_PATH=(
        Path(value)
        if (value := getenv("FFMPEG_PATH", "").strip())
        else None
    ),
    DEBUG_ARTIFACTS=getenv("DEBUG_ARTIFACTS", "true"),
    DEBUG_DIR=Path(getenv("DEBUG_DIR", "outputs/debug")),
    PIPELINE_VERSION=getenv("PIPELINE_VERSION", "v2").strip().lower(),
    V2_MAX_REPAIR_ATTEMPTS=getenv("V2_MAX_REPAIR_ATTEMPTS", "2"),
    V2_ENABLE_MULTIMODAL=getenv("V2_ENABLE_MULTIMODAL", "false").lower()
    in {"1", "true", "yes", "on"},
)
