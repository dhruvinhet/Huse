"""Tests for Edge-TTS narration generation and MP3 metadata."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core import audio_manager
from app.core.audio_manager import AudioGenerationError, AudioManager
from app.models.audio import AudioMetadata
from app.models.scene import Scene
from app.models.script import Script


def narrated_script() -> Script:
    """Create a two-scene script with narration for audio tests."""

    return Script(
        title="Sample",
        topic="Testing",
        total_duration=10,
        scenes=[
            Scene(
                scene_number=1,
                title="First",
                narration="First scene narration.",
                estimated_duration=5,
                visuals=[],
            ),
            Scene(
                scene_number=2,
                title="Second",
                narration="Second scene narration.",
                estimated_duration=5,
                visuals=[],
            ),
        ],
    )


def configure_audio_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point narration output to an isolated temporary folder."""

    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        audio_manager,
        "settings",
        SimpleNamespace(
            OUTPUT_DIR=output_dir,
            TEMP_DIR=tmp_path / "temp",
        ),
    )
    return output_dir / "audio" / "narration.mp3"


def test_edge_tts_is_mocked_and_metadata_is_returned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined narration is saved and returned as validated metadata."""

    audio_path = configure_audio_output(tmp_path, monkeypatch)
    def make_communicator(text: str, _voice: str) -> MagicMock:
        communicator = MagicMock()
        communicator.save_sync.side_effect = (
            lambda path: Path(path).write_bytes(text.encode("utf-8"))
        )
        return communicator

    constructor = MagicMock(side_effect=make_communicator)
    monkeypatch.setattr(audio_manager.edge_tts, "Communicate", constructor)

    def mp3_metadata(path: Path) -> SimpleNamespace:
        durations = {
            "scene_0001.mp3": 1.25,
            "scene_0002.mp3": 2.0,
        }
        return SimpleNamespace(
            info=SimpleNamespace(
                length=durations[Path(path).name],
                sample_rate=24000,
            )
        )

    monkeypatch.setattr(
        audio_manager,
        "MP3",
        MagicMock(side_effect=mp3_metadata),
    )

    metadata = AudioManager().generate(narrated_script())

    assert isinstance(metadata, AudioMetadata)
    assert metadata.duration == 3.25
    assert metadata.sample_rate == 24000
    assert metadata.voice == "en-US-AriaNeural"
    assert metadata.file_path == audio_path.as_posix()
    assert [scene.duration for scene in metadata.scenes] == [1.25, 2.0]
    assert metadata.scenes[1].start_time == 1.25
    assert metadata.scenes[1].end_time == 3.25
    assert [call.args[0] for call in constructor.call_args_list] == [
        "First scene narration.",
        "Second scene narration.",
    ]
    assert all(
        call.args[1] == "en-US-AriaNeural"
        for call in constructor.call_args_list
    )
    assert audio_path.read_bytes() == (
        b"First scene narration.Second scene narration."
    )
    manifest_path = audio_path.with_name("audio_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["duration"] == 3.25
    assert len(manifest["scenes"]) == 2


def test_empty_script_narration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A script without scenes is rejected before Edge-TTS is called."""

    constructor = MagicMock()
    monkeypatch.setattr(audio_manager.edge_tts, "Communicate", constructor)
    script = Script(
        title="Empty",
        topic="Testing",
        total_duration=1,
        scenes=[],
    )

    with pytest.raises(ValueError, match="narration text"):
        AudioManager().generate(script)

    constructor.assert_not_called()


def test_edge_tts_failure_is_translated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge-TTS failures become meaningful AudioGenerationError instances."""

    configure_audio_output(tmp_path, monkeypatch)
    communicator = MagicMock()
    communicator.save_sync.side_effect = RuntimeError("service unavailable")
    monkeypatch.setattr(
        audio_manager.edge_tts,
        "Communicate",
        MagicMock(return_value=communicator),
    )

    with pytest.raises(AudioGenerationError, match="service unavailable"):
        AudioManager().generate(narrated_script())
