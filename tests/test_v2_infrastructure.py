"""Infrastructure, schema, knowledge, and compatibility tests for V2."""

import json
from pathlib import Path

from app.adapters.legacy import LegacyScriptToStoryboardAdapter
from app.domain.lesson import LessonPlan
from app.domain.schemas import SCHEMA_MODELS, export_schemas, schema
from app.knowledge import JsonVisualKnowledgeBase
from app.models.scene import Scene, VisualInstruction
from app.models.script import Script
from app.observability import ArtifactCheckpointStore
from app.templates import TemplateRegistry, builtin_templates
from app.utils.debug_recorder import DebugRecorder
from tests.test_domain_v2 import concept_graph


def test_checkpoint_store_round_trips_and_detects_schema(tmp_path: Path) -> None:
    """Content-addressed checkpoints reload through their declared contract."""

    store = ArtifactCheckpointStore(tmp_path)
    lesson = LessonPlan(
        title="Test",
        summary="Test lesson",
        concept_graph=concept_graph(),
    )
    entry = store.write("Plan Lesson", lesson, "test")

    loaded = store.load("Plan Lesson", LessonPlan)
    assert loaded == lesson
    assert store.has("Plan Lesson")
    assert entry.content_hash
    assert json.loads((tmp_path / "artifact_manifest.json").read_text())[0][
        "producer"
    ] == "test"


def test_schema_registry_exports_every_contract(tmp_path: Path) -> None:
    """Committed contract names generate valid JSON Schema documents."""

    paths = export_schemas(tmp_path)
    assert len(paths) == len(SCHEMA_MODELS)
    assert schema("storyboard-2.0")["$id"] == "whiteboard/storyboard-2.0"
    assert all(json.loads(path.read_text())["type"] == "object" for path in paths)


def test_knowledge_base_accumulates_reviewed_evidence(tmp_path: Path) -> None:
    """Reviewed outcomes improve evidence without unreviewed self-learning."""

    store = JsonVisualKnowledgeBase(tmp_path / "knowledge.json")
    first = store.record_outcome(
        ["transformer", "attention"],
        "Reveal components and pulse the active path.",
        0.8,
        template_id="transformer_block.v1",
    )
    second = store.record_outcome(
        ["attention", "transformer"],
        "Reveal components and pulse the active path.",
        1.0,
        template_id="transformer_block.v1",
    )
    assert first.record_id == second.record_id
    assert second.review_count == 2
    assert second.quality_total == 1.8


def test_topic_template_library_contains_requested_domains() -> None:
    """The initial library covers algorithms, AI, systems, and networking."""

    registry = TemplateRegistry(builtin_templates())
    expected = {
        "binary_search.v1",
        "quick_sort.v1",
        "merge_sort.v1",
        "dfs.v1",
        "bfs.v1",
        "transformer_block.v1",
        "cnn.v1",
        "rnn.v1",
        "tcp_handshake.v1",
        "rest_api.v1",
        "microservices.v1",
        "database_index.v1",
        "os_scheduling.v1",
        "memory_allocation.v1",
        "hash_map.v1",
        "blockchain.v1",
        "authentication.v1",
        "system_design.v1",
    }
    assert expected.issubset(registry.template_ids())


def test_legacy_script_adapter_never_creates_empty_visuals() -> None:
    """V1 primitives receive explicit legacy semantics during migration."""

    script = Script(
        title="Legacy",
        topic="Legacy",
        total_duration=3,
        scenes=[
            Scene(
                scene_number=1,
                title="Flow",
                narration="Data flows.",
                estimated_duration=3,
                visuals=[
                    VisualInstruction(
                        type="box",
                        content="box",
                        position="center",
                        animation="draw",
                    )
                ],
            )
        ],
    )
    board = LegacyScriptToStoryboardAdapter().convert(script)
    created = board.beats[0].operations[0].arguments["objects"][0]
    assert created["content"]["label"] == "Box"
    assert created["semantic_role"] == "legacy_box"


def test_debug_recorder_atomically_finalizes_run_status(tmp_path: Path) -> None:
    """A completed result cannot coexist with a stale running status."""

    recorder = DebugRecorder(enabled=True, root_dir=tmp_path)
    run_dir = recorder.start_run("Status test")
    recorder.finish_run("complete", 1.0, {"stage": 0.5})
    assert run_dir is not None
    assert json.loads((run_dir / "run.json").read_text())["status"] == "complete"
    assert json.loads((run_dir / "result.json").read_text())["status"] == "complete"
