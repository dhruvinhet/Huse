"""Rule-based selection of topic-appropriate teaching rhetoric."""

import re

from app.domain.generation import AudienceProfile
from app.domain.lesson import ConceptRelation, LessonPlan
from app.domain.pedagogy import PedagogyMode, PedagogyPlan, PedagogyShot
from app.planning.graph_semantics import CAUSAL_RELATIONS, analyze_graph


_ShotDefinition = tuple[str, str, str, str, str]


class PedagogyRouter:
    """Choose a distinct shot grammar from stable lesson semantics."""

    _KEYWORDS: tuple[tuple[PedagogyMode, frozenset[str]], ...] = (
        (PedagogyMode.CONCEPT_OVERVIEW, frozenset(
            {"define", "definition", "overview", "introduction", "fundamentals", "basics"}
        )),
        (PedagogyMode.MISCONCEPTION_CORRECTION, frozenset(
            {"misconception", "myth", "mistake", "wrong", "confuse"}
        )),
        (PedagogyMode.ANALOGY, frozenset(
            {"analogy", "metaphor", "intuitive", "like", "imagine"}
        )),
        (PedagogyMode.CHRONOLOGICAL, frozenset({
            "history", "historical", "timeline", "chronology", "revolution",
            "war", "evolution", "era", "century",
        })),
        (PedagogyMode.PROOF_DERIVATION, frozenset({
            "proof", "prove", "derive", "derivation", "theorem", "identity",
            "equation",
        })),
        (PedagogyMode.CODE_EXECUTION, frozenset({
            "code", "program", "function", "loop", "variable", "debug",
            "recursion", "execute", "runtime",
        })),
        (PedagogyMode.WORKED_EXAMPLE, frozenset({
            "calculate", "calculation", "interest", "search", "sort",
            "algorithm", "solve", "example", "percentage", "loan",
        })),
        (PedagogyMode.COMPARISON, frozenset({
            "compare", "comparison", "versus", "difference", "tradeoff",
            "min", "max", "advantages", "disadvantages",
        })),
        (PedagogyMode.SIMULATION, frozenset({
            "probability", "random", "simulation", "feedback", "dynamic",
            "population", "growth", "decay",
        })),
        (PedagogyMode.SPATIAL_ANATOMY, frozenset({
            "anatomy", "structure", "component", "architecture", "layer",
            "cell", "atom", "organ", "circuit", "parts",
        })),
    )

    _GRAMMARS: dict[
        PedagogyMode,
        tuple[str, tuple[_ShotDefinition, ...]],
    ] = {
        PedagogyMode.CONCEPT_OVERVIEW: (
            "Define the idea, establish its boundary, and connect it to concrete uses.",
            (
                ("definition_frame", "introduce", "Show the concept and its defining boundary.", "Give a direct definition and say what problem the concept addresses.", "fit"),
                ("core_ideas", "demonstrate", "Organize the essential ideas without implying a mandatory process.", "Explain the essential ideas and what distinguishes each one.", "focus"),
                ("concrete_uses", "demonstrate", "Connect the ideas to representative outcomes or uses.", "Ground the definition in concrete outcomes without inventing a causal chain.", "focus"),
                ("overview_synthesis", "summarize", "Return to the complete concept map.", "Restate the definition and connect the important ideas.", "hold"),
            ),
        ),
        PedagogyMode.CONCEPT_SET: (
            "Teach co-equal members as a collection without inventing sequence.",
            (
                ("collection_frame", "introduce", "Show the complete collection and its organizing question.", "Orient the viewer to what unites these ideas without implying that one causes the next.", "fit"),
                ("member_focus_a", "demonstrate", "Focus the first members with their own defining evidence.", "Explain each focused member independently with a concrete implication or example.", "focus"),
                ("member_focus_b", "demonstrate", "Focus the remaining members and distinguish their scope.", "Explain the remaining members and how their conditions differ from the earlier ones.", "focus"),
                ("collection_synthesis", "summarize", "Return to all members as one coordinated framework.", "Summarize what unites the collection and when each member applies.", "hold"),
            ),
        ),
        PedagogyMode.MECHANISM_FIRST: (
            "Explain causal structure from whole system to consequence.",
            (
                ("system_view", "introduce", "Show the complete system boundary.", "Name the system goal and orient the viewer.", "fit"),
                ("causal_steps", "transform", "Reveal the mechanism in causal order.", "Explain why each visible change causes the next.", "focus"),
                ("relationships", "connect", "Make dependencies and flows explicit.", "State the relationship, not merely the two labels.", "focus"),
                ("consequence", "summarize", "Return to the whole mechanism and outcome.", "Connect the mechanism to its final consequence.", "hold"),
            ),
        ),
        PedagogyMode.WORKED_EXAMPLE: (
            "Carry one concrete case from setup to verification.",
            (
                ("problem_setup", "introduce", "Show one concrete problem and known values.", "State what is given and what must be found.", "fit"),
                ("initial_state", "demonstrate", "Mark the initial state and first decision.", "Explain why the first operation is valid.", "focus"),
                ("step_through", "transform", "Advance through visible state changes.", "Narrate each change using the current values.", "focus"),
                ("result_check", "summarize", "Verify the result against the problem.", "State the answer and the reusable method.", "hold"),
            ),
        ),
        PedagogyMode.MISCONCEPTION_CORRECTION: (
            "Replace a tempting wrong model with a tested correct model.",
            (
                ("tempting_claim", "introduce", "Display the common mistaken claim.", "Explain why the claim initially feels plausible.", "fit"),
                ("counterexample", "compare", "Place a counterexample beside the claim.", "Identify the observation the claim cannot explain.", "focus"),
                ("correct_model", "transform", "Replace the claim with the correct relation.", "Explain the corrected mechanism or rule.", "focus"),
                ("diagnostic_rule", "summarize", "Show a quick check for avoiding the mistake.", "Give a reusable diagnostic rule.", "hold"),
            ),
        ),
        PedagogyMode.ANALOGY: (
            "Map a familiar system to the target while exposing limits.",
            (
                ("familiar_scene", "introduce", "Show the familiar source analogy.", "Describe only the familiar behavior first.", "fit"),
                ("mapping", "connect", "Map source parts to target concepts.", "Name each correspondence explicitly.", "focus"),
                ("analogy_limit", "compare", "Mark where the analogy stops matching.", "State one limitation that prevents transfer errors.", "focus"),
                ("target_transfer", "summarize", "Return to the target without analogy labels.", "Explain the target idea in its own terms.", "hold"),
            ),
        ),
        PedagogyMode.PROOF_DERIVATION: (
            "Build a justified derivation from givens to conclusion.",
            (
                ("givens", "introduce", "Show givens, definitions, and target.", "State assumptions and the result to establish.", "fit"),
                ("derivation", "transform", "Transform one expression at a time.", "Justify every visible transformation.", "focus"),
                ("key_step", "emphasize", "Highlight the non-obvious inference.", "Explain why the key inference is permitted.", "focus"),
                ("conclusion", "summarize", "Align the result with the target.", "State what was proved and under which assumptions.", "hold"),
            ),
        ),
        PedagogyMode.CHRONOLOGICAL: (
            "Explain change through context, turning points, and consequences.",
            (
                ("context", "introduce", "Establish starting conditions on a timeline.", "Describe conditions before the first event.", "fit"),
                ("turning_points", "connect", "Reveal ordered turning points.", "Explain how each event changes what follows.", "focus"),
                ("consequences", "transform", "Connect events to their effects.", "Separate sequence from causal consequence.", "focus"),
                ("historical_synthesis", "summarize", "Show the complete chronology.", "Summarize the transformation across time.", "hold"),
            ),
        ),
        PedagogyMode.COMPARISON: (
            "Evaluate alternatives against shared criteria.",
            (
                ("comparison_frame", "introduce", "Place alternatives in one frame.", "Name the comparison question and criteria.", "fit"),
                ("side_by_side", "compare", "Align corresponding features.", "Contrast the same feature before moving on.", "focus"),
                ("tradeoffs", "compare", "Highlight gains, costs, and conditions.", "Explain when each alternative is preferable.", "focus"),
                ("decision_rule", "summarize", "End with a criterion-based choice rule.", "Give a conditional rule, not a universal winner.", "hold"),
            ),
        ),
        PedagogyMode.SIMULATION: (
            "Vary state or parameters and observe resulting behavior.",
            (
                ("initial_conditions", "introduce", "Show initial state and parameters.", "Define what changes and what is measured.", "fit"),
                ("run_model", "transform", "Advance the system one step at a time.", "Narrate the state update and its cause.", "focus"),
                ("vary_parameter", "demonstrate", "Change one parameter only.", "Compare the new trajectory with the baseline.", "focus"),
                ("observed_pattern", "summarize", "Align the resulting patterns.", "State the observed relationship and its limits.", "hold"),
            ),
        ),
        PedagogyMode.SPATIAL_ANATOMY: (
            "Move from whole structure to parts, relations, and function.",
            (
                ("whole_view", "introduce", "Show the intact whole with orientation.", "Name the whole and its main function.", "fit"),
                ("inspect_parts", "demonstrate", "Reveal parts by spatial region.", "Explain each part where it appears.", "focus"),
                ("part_relations", "connect", "Show how neighboring parts interact.", "Describe the functional relationship between parts.", "focus"),
                ("function_summary", "summarize", "Return to the active whole.", "Connect structure to overall function.", "hold"),
            ),
        ),
        PedagogyMode.CODE_EXECUTION: (
            "Trace executable state rather than static code description.",
            (
                ("input_state", "introduce", "Show code, input, and initial state.", "State the input and initial execution point.", "fit"),
                ("execute_step", "transform", "Advance the active statement.", "Explain the statement and exact state change.", "focus"),
                ("inspect_control", "demonstrate", "Expose control-flow decisions.", "Explain why control moves to the next location.", "focus"),
                ("output_trace", "summarize", "Connect final state to output.", "State the output and execution pattern.", "hold"),
            ),
        ),
    }

    def route(
        self,
        lesson: LessonPlan,
        audience: AudienceProfile,
    ) -> PedagogyPlan:
        """Return a deterministic mode and its required shots."""

        source_text = self._lesson_text(lesson, audience)
        terms = set(re.findall(r"[a-z0-9]+", source_text))
        mode = self._relation_mode(lesson)
        scores: dict[PedagogyMode, int] = {candidate: 0 for candidate, _ in self._KEYWORDS}
        for candidate, keywords in self._KEYWORDS:
            scores[candidate] = len(terms & keywords)
        structure = analyze_graph(lesson.concept_graph)
        dynamic_relations = set(CAUSAL_RELATIONS)
        relation_set = {edge.relation for edge in lesson.concept_graph.edges}
        mechanism_score = 2 * len(relation_set & dynamic_relations)
        if "how" in terms and terms & {
            "work", "works", "working", "mechanism", "flow", "flows",
            "change", "changes", "transform", "transforms",
        }:
            mechanism_score += 3
        if mechanism_score:
            scores[PedagogyMode.MECHANISM_FIRST] = mechanism_score
        if re.search(r"\bwhat\s+(?:is|are)\b", source_text):
            scores[PedagogyMode.CONCEPT_OVERVIEW] = max(
                8,
                scores.get(PedagogyMode.CONCEPT_OVERVIEW, 0),
            )
        if (
            structure.peer_collection
            and not structure.is_process
            and not structure.contrast_edges
        ):
            scores[PedagogyMode.CONCEPT_SET] = 9
        best_score = max(scores.values(), default=0)
        if best_score:
            mode = max(
                scores,
                key=lambda item: (
                    scores[item],
                    item is PedagogyMode.MECHANISM_FIRST,
                    item.value,
                ),
            )
        rationale, definitions = self._GRAMMARS[mode]
        confidence = (
            "Matched explicit lesson terms."
            if best_score
            else "Selected from concept relationships and mechanism structure."
        )
        return PedagogyPlan(
            mode=mode,
            rationale=f"{rationale} {confidence}",
            shots=[
                PedagogyShot(
                    shot_id=shot_id,
                    purpose=purpose,
                    visual_obligation=visual,
                    narration_obligation=narration,
                    camera_operation=camera,
                )
                for shot_id, purpose, visual, narration, camera in definitions
            ],
        )

    @staticmethod
    def _lesson_text(
        lesson: LessonPlan,
        audience: AudienceProfile,
    ) -> str:
        """Return normalized request and lesson language with phrase order intact."""

        return " ".join(
            [
                lesson.title,
                lesson.summary,
                audience.learning_goal,
                *lesson.concept_graph.objectives,
                *[
                    f"{node.label} {node.definition} "
                    + " ".join(node.visual_affordances)
                    for node in lesson.concept_graph.nodes
                ],
            ]
        ).lower()

    @staticmethod
    def _relation_mode(lesson: LessonPlan) -> PedagogyMode:
        relations = {edge.relation for edge in lesson.concept_graph.edges}
        structure = analyze_graph(lesson.concept_graph)
        if ConceptRelation.CONTRASTS_WITH in relations:
            return PedagogyMode.COMPARISON
        if relations and relations == {ConceptRelation.PART_OF}:
            return PedagogyMode.SPATIAL_ANATOMY
        if structure.peer_collection and not structure.is_process:
            return PedagogyMode.CONCEPT_SET
        if structure.is_process:
            return PedagogyMode.MECHANISM_FIRST
        return PedagogyMode.CONCEPT_OVERVIEW
