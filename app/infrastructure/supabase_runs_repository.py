"""Supabase persistence for video-generation job status."""

from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from app.config.settings import settings


class SupabaseConfigurationError(RuntimeError):
    """Raised when the backend has no usable Supabase configuration."""


class SupabaseRunsRepository:
    """Persist generation lifecycle records through Supabase's Data API."""

    def __init__(self) -> None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SECRET_KEY:
            raise SupabaseConfigurationError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY must be configured for "
                "backend database access."
            )
        self._client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SECRET_KEY,
        )

    def create(self, topic: str) -> str:
        """Create a queued generation and return its database ID."""
        response = (
            self._client.table("generation_runs")
            .insert({"topic": topic, "status": "queued"})
            .execute()
        )
        if not response.data:
            raise RuntimeError("Supabase did not return the created generation run.")
        return response.data[0]["id"]

    def mark_running(self, run_id: str) -> None:
        """Record that background generation has started."""
        self._update(run_id, {"status": "running"})

    def mark_completed(self, run_id: str, output_path: str) -> None:
        """Record successful completion and the generated video location."""
        self._update(
            run_id,
            {
                "status": "completed",
                "output_path": output_path,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def mark_failed(self, run_id: str, message: str) -> None:
        """Record a bounded, user-safe failure message."""
        self._update(
            run_id,
            {
                "status": "failed",
                "error_message": message[:2000],
            },
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return one generation record, if it exists."""
        response = (
            self._client.table("generation_runs")
            .select("*")
            .eq("id", run_id)
            .execute()
        )
        return response.data[0] if response.data else None

    def _update(self, run_id: str, values: dict[str, Any]) -> None:
        (
            self._client.table("generation_runs")
            .update(values)
            .eq("id", run_id)
            .execute()
        )
