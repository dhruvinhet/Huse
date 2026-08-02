"""HTTP API for starting and inspecting video-generation jobs."""

from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.application.orchestrators import V2PipelineRunner
from app.domain.generation import AudienceProfile, GenerationRequest
from app.infrastructure.supabase_runs_repository import (
    SupabaseConfigurationError,
    SupabaseRunsRepository,
)


app = FastAPI(title="Whiteboard AI Video Generator", version="1.0.0")


class GenerationStartRequest(BaseModel):
    """Client input for a new educational video."""

    topic: str = Field(min_length=1, max_length=500)


def _repository() -> SupabaseRunsRepository:
    try:
        return SupabaseRunsRepository()
    except SupabaseConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def _generate(run_id: str, topic: str) -> None:
    """Run the long-lived V2 compilation after the HTTP response is returned."""

    repository = SupabaseRunsRepository()
    repository.mark_running(run_id)
    try:
        result = V2PipelineRunner().run(
            GenerationRequest(
                run_id=run_id,
                topic=topic,
                audience=AudienceProfile(learning_goal=f"Understand {topic}"),
            )
        )
        repository.mark_completed(run_id, result.output_file)
    except Exception as exc:
        repository.mark_failed(run_id, str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight process health response."""

    return {"status": "ok"}


@app.post("/generations", status_code=status.HTTP_202_ACCEPTED)
def start_generation(
    request: GenerationStartRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Create a persisted job and begin generation asynchronously."""

    repository = _repository()
    run_id = repository.create(request.topic.strip())
    background_tasks.add_task(_generate, run_id, request.topic.strip())
    return {"id": run_id, "status": "queued"}


@app.get("/generations/{run_id}")
def get_generation(run_id: UUID) -> dict[str, object]:
    """Return the persisted state of one generation job."""

    record = _repository().get(str(run_id))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    return record
