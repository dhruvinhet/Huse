"""Application bootstrap for the Whiteboard AI Video Generator."""

from loguru import logger

from app.config.settings import settings
from app.utils.logger import initialize_logger


def main() -> None:
    """Initialize the project foundation and report successful startup."""

    initialize_logger(settings.LOG_LEVEL)
    logger.info("Whiteboard AI Video Generator initialized successfully.")


if __name__ == "__main__":
    main()
