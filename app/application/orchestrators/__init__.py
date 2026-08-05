"""Versioned pipeline orchestrators, loaded on demand."""


def __getattr__(name: str) -> object:
    if name not in {"QualityGateError", "V2PipelineRunner"}:
        raise AttributeError(name)
    from app.application.orchestrators.pipeline_v2 import (
        QualityGateError,
        V2PipelineRunner,
    )
    return {"QualityGateError": QualityGateError, "V2PipelineRunner": V2PipelineRunner}[name]


__all__ = ["QualityGateError", "V2PipelineRunner"]
