"""Public use cases, loaded only when requested."""


def __getattr__(name: str) -> object:
    if name != "VideoGenerationService":
        raise AttributeError(name)
    from app.application.use_cases.generate_video import VideoGenerationService
    return VideoGenerationService


__all__ = ["VideoGenerationService"]
