"""Run-scoped structured debug artifacts for pipeline observability."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from loguru import logger

from app.config.settings import PROJECT_ROOT


class DebugRecorder:
    """Persist inspectable text, JSON, JSONL, and sample files per run."""

    def __init__(self, enabled: bool, root_dir: Path) -> None:
        """Configure debug recording without creating a run yet."""

        self.enabled = enabled
        self._root_dir = self._working_path(root_dir)
        self._run_dir: Path | None = None
        self._log_sink_id: int | None = None
        self._lock = Lock()

    @property
    def run_dir(self) -> Path | None:
        """Return the active run directory when debugging is enabled."""

        return self._run_dir

    def start_run(self, topic: str) -> Path | None:
        """Create a unique run folder and human-readable artifact index."""

        if not self.enabled:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{timestamp}_{uuid4().hex[:8]}"
        self._run_dir = self._root_dir / "runs" / run_id
        self._run_dir.mkdir(parents=True, exist_ok=False)
        self._close_log_sink()
        self._log_sink_id = logger.add(
            self._run_dir / "pipeline.log",
            level="DEBUG",
            rotation=None,
            enqueue=True,
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
        )
        self.write_json(
            "run.json",
            {
                "run_id": run_id,
                "topic": topic,
                "started_at": datetime.now().astimezone().isoformat(),
                "status": "running",
            },
        )
        self.write_text(
            "README.txt",
            (
                "Whiteboard Pipeline Debug Bundle\n\n"
                "llm/ contains the exact prompt, raw response, and script.\n"
                "pipeline.log contains every DEBUG/INFO/WARNING/ERROR event for this run.\n"
                "tts/ contains narration text and measured scene timings.\n"
                "pipeline/ contains assets, scene graph, timelines, and manifest.\n"
                "frames/frame_trace.jsonl contains one JSON record per frame.\n"
                "frames/samples/ contains representative rendered PNGs.\n"
                "ffmpeg/ contains the command and verified output stream data.\n"
                "The Gemini API key is never recorded.\n"
            ),
        )
        self._root_dir.mkdir(parents=True, exist_ok=True)
        latest_path = self._root_dir / "latest_run.json"
        latest_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "path": self._run_dir.as_posix(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Pipeline debug artifacts: {}.", self._run_dir)
        return self._run_dir

    def finish_run(
        self,
        status: str,
        execution_time: float,
        stage_timings: dict[str, float],
        error: BaseException | None = None,
    ) -> None:
        """Write final status while retaining artifacts from failed runs."""

        if not self.enabled or self._run_dir is None:
            return
        payload: dict[str, Any] = {
            "status": status,
            "finished_at": datetime.now().astimezone().isoformat(),
            "execution_time_seconds": execution_time,
            "stage_timings_seconds": stage_timings,
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        try:
            logger.info(
                "Pipeline run finished (status={}, execution_time_seconds={:.3f}).",
                status,
                execution_time,
            )
            self.write_json("result.json", payload)
            run_path = self._run_dir / "run.json"
            try:
                run_payload = json.loads(run_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                run_payload = {}
            run_payload.update(
                {
                    "status": status,
                    "finished_at": payload["finished_at"],
                }
            )
            temporary_path = run_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(run_payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            temporary_path.replace(run_path)
        finally:
            self._close_log_sink()

    def write_text(self, relative_path: str, content: str) -> None:
        """Write one UTF-8 text artifact when a run is active."""

        destination = self._destination(relative_path)
        if destination is None:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    def write_json(self, relative_path: str, value: Any) -> None:
        """Write one readable JSON artifact using safe string fallbacks."""

        self.write_text(
            relative_path,
            json.dumps(value, indent=2, ensure_ascii=False, default=str),
        )

    def append_jsonl(self, relative_path: str, value: Any) -> None:
        """Append one compact JSON record for streaming frame traces."""

        self.append_jsonl_many(relative_path, [value])

    def append_jsonl_many(
        self,
        relative_path: str,
        values: list[Any],
    ) -> None:
        """Append multiple JSON records with one filesystem operation."""

        destination = self._destination(relative_path)
        if destination is None or not values:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = "\n".join(
            json.dumps(value, ensure_ascii=False, default=str)
            for value in values
        )
        with self._lock, destination.open("a", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.write("\n")

    def copy_file(self, source: Path, relative_path: str) -> None:
        """Copy a representative artifact without loading it into memory."""

        destination = self._destination(relative_path)
        if destination is None:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def _destination(self, relative_path: str) -> Path | None:
        """Resolve an internal relative artifact path for the active run."""

        if not self.enabled or self._run_dir is None:
            return None
        return self._run_dir / Path(relative_path)

    def _close_log_sink(self) -> None:
        """Flush and remove the active per-run Loguru sink."""

        if self._log_sink_id is not None:
            logger.remove(self._log_sink_id)
            self._log_sink_id = None

    @staticmethod
    def _working_path(configured_path: Path) -> Path:
        """Resolve configured relative paths against the project root."""

        if configured_path.is_absolute():
            return configured_path
        return PROJECT_ROOT / configured_path
