"""Tests for provider/estimated word-alignment provenance."""

from app.audio import ScenePhraseAligner
from app.benchmarking.metrics import collect_metrics
from app.domain.narration import NarrationPhrase, NarrationPlan
from app.models.audio import AudioMetadata, AudioWordTiming, SceneAudio


def test_mixed_word_timing_retains_per_word_source_and_confidence() -> None:
    """Mixed scene timing must not upgrade estimated words to provider confidence."""

    narration = NarrationPlan(
        phrases=[
            NarrationPhrase(phrase_id="p1", beat_id="b1", text="provider word"),
            NarrationPhrase(phrase_id="p2", beat_id="b2", text="estimated word"),
        ]
    )
    audio = AudioMetadata(
        file_path="mixed.mp3",
        duration=2,
        sample_rate=24000,
        voice="voice",
        word_timing_source="mixed",
        scenes=[
            SceneAudio(
                scene_number=1, duration=1, text="provider word",
                start_time=0, end_time=1,
            ),
            SceneAudio(
                scene_number=2, duration=1, text="estimated word",
                start_time=1, end_time=2,
            ),
        ],
        words=[
            AudioWordTiming(
                scene_number=1, text="provider", start_time=0, end_time=0.5,
                timing_source="provider",
            ),
            AudioWordTiming(
                scene_number=2, text="estimated", start_time=1, end_time=1.5,
                timing_source="estimated",
            ),
        ],
    )

    aligned = ScenePhraseAligner().align(narration, audio)

    assert [word.timing_source for word in aligned.words] == [
        "provider", "estimated"
    ]
    assert [word.confidence for word in aligned.words] == [1.0, 0.35]
    metrics = collect_metrics(narration=narration, alignment=aligned)
    assert metrics.provider_timing_coverage == 0.25
    assert metrics.estimated_timing_coverage == 0.25
    assert metrics.mean_word_timing_confidence == 0.675
