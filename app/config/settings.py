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

    AI_PROVIDER: str
    GEMINI_API_KEY: str
    GEMINI_MODEL: str
    NVIDIA_API_KEY: str
    NVIDIA_MODEL: str
    NVIDIA_VISION_MODEL: str
    NVIDIA_BASE_URL: str
    NVIDIA_MAX_TOKENS: int
    NVIDIA_TIMEOUT_SECONDS: float
    OUTPUT_DIR: Path
    TEMP_DIR: Path
    LOG_LEVEL: str
    FFMPEG_PATH: Path | None
    DEBUG_ARTIFACTS: bool
    DEBUG_FRAME_TRACE_FULL: bool
    DEBUG_DIR: Path
    PIPELINE_VERSION: str
    V2_MAX_REPAIR_ATTEMPTS: int
    V2_ENABLE_MULTIMODAL: bool
    SUPABASE_URL: str
    SUPABASE_SECRET_KEY: str


settings = Settings(
    AI_PROVIDER=getenv("AI_PROVIDER", "gemini").strip().lower(),
    GEMINI_API_KEY=getenv("GEMINI_API_KEY", ""),
    GEMINI_MODEL=getenv("GEMINI_MODEL", "gemini-3.5-flash").strip(),
    NVIDIA_API_KEY=getenv("NVIDIA_API_KEY", "").strip(),
    NVIDIA_MODEL=getenv(
        "NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"
    ).strip(),
    NVIDIA_VISION_MODEL=getenv(
        "NVIDIA_VISION_MODEL", "google/gemma-3n-e4b-it"
    ).strip(),
    NVIDIA_BASE_URL=getenv(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    ).strip().rstrip("/"),
    NVIDIA_MAX_TOKENS=getenv("NVIDIA_MAX_TOKENS", "16384"),
    NVIDIA_TIMEOUT_SECONDS=getenv("NVIDIA_TIMEOUT_SECONDS", "600"),
    OUTPUT_DIR=Path(getenv("OUTPUT_DIR", "outputs")),
    TEMP_DIR=Path(getenv("TEMP_DIR", "temp")),
    LOG_LEVEL=getenv("LOG_LEVEL", "INFO"),
    FFMPEG_PATH=(
        Path(value)
        if (value := getenv("FFMPEG_PATH", "").strip())
        else None
    ),
    DEBUG_ARTIFACTS=getenv("DEBUG_ARTIFACTS", "false"),
    DEBUG_FRAME_TRACE_FULL=getenv("DEBUG_FRAME_TRACE_FULL", "false").lower()
    in {"1", "true", "yes", "on"},
    DEBUG_DIR=Path(getenv("DEBUG_DIR", "outputs/debug")),
    PIPELINE_VERSION=getenv("PIPELINE_VERSION", "v2").strip().lower(),
    V2_MAX_REPAIR_ATTEMPTS=getenv("V2_MAX_REPAIR_ATTEMPTS", "2"),
    V2_ENABLE_MULTIMODAL=getenv("V2_ENABLE_MULTIMODAL", "false").lower()
    in {"1", "true", "yes", "on"},
    SUPABASE_URL=getenv("SUPABASE_URL", "").strip(),
    SUPABASE_SECRET_KEY=getenv("SUPABASE_SECRET_KEY", "").strip(),
)
