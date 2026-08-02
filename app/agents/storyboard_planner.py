"""Specialized AI visual-storyboard planner."""

from app.agents.base import StructuredGeminiAgent
from app.domain.lesson import LessonPlan
from app.domain.storyboard import Storyboard
from app.domain.strategy import TemplateMatch, VisualStrategy
from app.domain.quality import QualityReport
from app.services.gemini_client import GeminiClient


class GeminiStoryboardPlanner:
    """Convert a lesson into persistent semantic visual beats."""

    def __init__(self, client: GeminiClient, max_attempts: int = 3) -> None:
        """Initialize the structured storyboard-planning agent."""

        self._agent = StructuredGeminiAgent(
            client,
            Storyboard,
            "Visual Storyboard Planner",
            max_attempts,
        )

    def plan(
        self,
        lesson: LessonPlan,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
    ) -> Storyboard:
        """Create progressive visual operations without coordinates."""

        instructions = (
            "Create a storyboard that teaches through evolving visuals. Use "
            "persistent object IDs and semantic object kinds. Never output x, y, "
            "width, height, or other pixel coordinates. Each beat must introduce, "
            "change, connect, annotate, or emphasize meaningful objects and must "
            "include phrase_intent for the narration writer. Prefer matched "
            "templates and recognized semantic kinds over generic shapes. Every "
            "connector needs source_id and target_id in content. Progressively "
            "disclose outlines, labels, connections, highlights, and conclusions. "
            "Assign every beat a purpose: introduce, demonstrate, compare, "
            "transform, connect, emphasize, or summarize. Build a teaching arc "
            "that normally introduces the idea, demonstrates or transforms it, "
            "connects relationships, and ends with a compact visual summary. "
            "Use comparison, timeline, cycle, cause-effect, chart, equation, or "
            "architecture semantics when they teach the concept more clearly. "
            "Create operations must include an arguments.objects array containing "
            "complete VisualObjectSpec-shaped definitions matching target_ids; "
            "never emit a create operation without that array. "
            "Follow this object-lifecycle rule exactly: an update, move, resize, "
            "highlight, dim, morph, connect, disconnect, show, hide, group, or "
            "erase operation may target only an object already present in "
            "initial_objects or created by an earlier beat. The safest valid arc "
            "is to create all reusable objects in the first beat, then reference "
            "only those existing IDs in later beats. Never reference an object "
            "before its create operation, and never reuse a retired ID. "
            "If an object already exists, update or highlight it; never emit a "
            "second create operation for that object ID or any child ID. "
            "Keep document_id and initial_objects at the storyboard top level; do not "
            "place initial_objects inside a beat. "
            "Do not write final narration and do not restart the canvas between beats. "
            "Keep the response compact: use at most 4 beats, one operation per "
            "beat unless a second is essential, and terse strings. Omit optional "
            "attention, camera_intent, children, constraints, and final summary "
            "entries unless they are necessary for teaching. Create only the "
            "minimal required object fields and avoid verbose content. Every beat "
            "must contain at least one operation."
        )
        payload = {
            "lesson": lesson.model_dump(mode="json"),
            "strategies": [item.model_dump(mode="json") for item in strategies],
            "template_matches": [item.model_dump(mode="json") for item in templates],
        }
        return self._agent.generate(instructions, payload)

    def repair(
        self,
        lesson: LessonPlan,
        previous: Storyboard,
        report: QualityReport,
        strategies: list[VisualStrategy],
        templates: list[TemplateMatch],
    ) -> Storyboard:
        """Repair only defects identified by a structured quality report."""

        instructions = (
            "Repair the previous storyboard using every actionable quality "
            "finding. Preserve correct concepts and stable object IDs whenever "
            "possible. Do not output coordinates. Remove placeholders, empty "
            "decorative shapes, dangling references, unexplained static spans, "
            "and missing high-importance concepts. Return the complete repaired "
            "storyboard, not a patch."
        )
        payload = {
            "lesson": lesson.model_dump(mode="json"),
            "previous_storyboard": previous.model_dump(mode="json"),
            "quality_report": report.model_dump(mode="json"),
            "strategies": [item.model_dump(mode="json") for item in strategies],
            "template_matches": [item.model_dump(mode="json") for item in templates],
        }
        return self._agent.generate(instructions, payload)
