"""End-to-end rendered-video acceptance for binary-tree obligations."""

from pathlib import Path

import pytest
from PIL import Image, ImageChops

from app.camera import SemanticCameraPlanner
from app.domain.assets import ResolvedAssetSet
from app.domain.generation import AudienceProfile
from app.domain.layout import Viewport
from app.domain.lesson import ConceptGraph, ConceptNode, LessonPlan
from app.domain.narration import AlignedAudio, PhraseTiming
from app.domain.rendering import RenderJob
from app.layout import HierarchicalLayoutEngine
from app.motion import SemanticAnimationPlanner
from app.planning import PedagogyRouter, TemplateCompiler
from app.rendering import SemanticFrameRenderer
from app.state import VisualStateTransitionEngine
from app.templates import TemplateRegistry
from app.templates.builtins import builtin_templates
from app.timeline import PhraseManifestBuilder


BINARY_TREE_PROMPTS = [
    "Explain a binary tree with root, children, leaves, height, and traversal",
    "Teach binary-tree structure and visit its nodes in order",
    "Show how a binary tree branches and how traversal proceeds",
    "Introduce binary trees using a concrete connected example",
    "Demonstrate binary tree levels, subtrees, and an ordered traversal",
]


def _lesson(objective: str) -> LessonPlan:
    labels = ["8", "4", "12", "2", "6", "10", "14"]
    nodes = [
        ConceptNode(
            concept_id=f"node_{index}",
            label=label,
            definition=f"Node {label} in the concrete binary tree.",
            importance=1,
            teaching_order=index,
            visual_affordances=["binary tree", "tree node"],
        )
        for index, label in enumerate(labels)
    ]
    return LessonPlan(
        title="Binary Tree",
        summary="A root branches to at most two children and ordered visits traverse the nodes.",
        concept_graph=ConceptGraph(
            objectives=[objective],
            nodes=nodes,
            teaching_sequence=[node.concept_id for node in nodes],
        ),
    )


@pytest.mark.parametrize("objective", BINARY_TREE_PROMPTS)
def test_five_binary_tree_prompts_render_real_video_with_topology_and_traversal(
    tmp_path: Path,
    objective: str,
) -> None:
    """Each accepted prompt produces an encoded nonblank tree/traversal video."""

    try:
        ffmpeg = SemanticFrameRenderer._find_ffmpeg()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    if ffmpeg is None:
        pytest.skip("FFmpeg is unavailable for rendered-video acceptance")

    lesson = _lesson(objective)
    audience = AudienceProfile(learning_goal=objective)
    pedagogy = PedagogyRouter().route(lesson, audience)
    registry = TemplateRegistry(builtin_templates())
    program = TemplateCompiler().compile(
        lesson,
        [],
        registry.match(lesson.concept_graph, audience),
        registry,
        pedagogy,
        target_duration=float(len(pedagogy.shots)),
        audience=audience,
    )
    assert program is not None
    board = program.storyboard
    route_beat = next(
        beat
        for beat in board.beats
        if any(action.action == "route" for action in beat.semantic_actions)
    )
    route = next(
        action for action in route_beat.semantic_actions if action.action == "route"
    )
    tree_root = next(
        root
        for beat in board.beats
        for operation in beat.operations
        for root in operation.arguments.get("objects", [])
        if isinstance(root, dict) and root.get("kind") == "tree"
    )
    nodes = [item for item in tree_root["children"] if item["kind"] == "tree_node"]
    edges = [item for item in tree_root["children"] if item["kind"] == "connector"]
    assert len(edges) == len(nodes) - 1
    assert len([route.source_id, *route.path_ids, route.target_id]) == len(nodes)

    phrases = []
    for index, beat in enumerate(board.beats):
        phrases.append(PhraseTiming(
            phrase_id=f"phrase_{index}",
            beat_id=beat.beat_id,
            audio_start=float(index),
            audio_end=float(index + 1),
        ))
    alignment = AlignedAudio(
        audio_path="binary_tree_acceptance.wav",
        duration=float(len(board.beats)),
        sample_rate=24000,
        phrases=phrases,
    )
    document = VisualStateTransitionEngine().materialize(board)
    assets = ResolvedAssetSet()
    layout = HierarchicalLayoutEngine().layout(
        document,
        assets,
        Viewport(width=640, height=360, margin=20),
    )
    motion = SemanticAnimationPlanner().plan(board, layout, alignment)
    camera = SemanticCameraPlanner().plan(board, layout, alignment)
    manifest = PhraseManifestBuilder().build(board, alignment, fps=4)
    output = tmp_path / str(abs(hash(objective)))
    result = SemanticFrameRenderer().render(RenderJob(
        run_id=f"binary_tree_{abs(hash(objective))}",
        document=document,
        assets=assets,
        layout=layout,
        motion=motion,
        camera=camera,
        manifest=manifest,
        output_folder=output.as_posix(),
        keep_frames=False,
    ))

    video = Path(result.video_stream_path or "")
    assert video.is_file()
    assert video.stat().st_size > 1_000
    assert result.total_frames == len(board.beats) * 4
    route_event = next(
        event
        for event in motion.events
        if event.parameters.get("semantic_action") == "route"
    )
    assert len(route_event.parameters["trajectories"]["__route__"]) == len(nodes)
    samples = [Image.open(path).convert("RGB") for path in result.sample_paths]
    assert samples
    white = Image.new("RGB", samples[0].size, "white")
    assert all(ImageChops.difference(sample, white).getbbox() for sample in samples)
