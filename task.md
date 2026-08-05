# Huse verified remediation backlog

This backlog is based on a finding-by-finding review of
`Huse_Source_Code_Reaudit_7ce0030.docx` against branch `Harsh---DB` at
commit `7ce00303c4cb5e8757a1bb89691416c909de7d83`.

## Baseline and evidence rules

- The audit baseline is commit `7ce0030`; T00 is checkpointed on top of it as
  commit `2f1005f`.
- The source inventory matches the report for the important code assets: 177
  Python files, 5,910 SVG assets, 39 built-in templates, and 42 registered
  renderer operators.
- The report's 188-test result is plausible for the clean commit. The T00
  checkpoint passes 191 tests.
- T00 preserved and committed these corrective files:
  - `app/quality/educational.py`
  - `app/layout/engine.py`
  - `tests/test_generalized_semantic_pipeline.py`
  - `tests/test_v2_layout_motion_quality.py`
- The report's ZIP-level total of 6,117 files, 80% coverage number, companion
  CSV data, sample-video observations, exact runtime numbers, and Simi output
  comparisons cannot be independently reproduced from the supplied DOCX and
  repository. Treat them as audit observations, not release evidence.
- Status labels below mean:
  - **Confirmed**: directly supported by the current source.
  - **Partly confirmed**: the gap exists, but the report omits or overstates
    existing behavior.
  - **Resolved**: the report accurately describes an implemented capability;
    no new task is needed unless a residual gap is named.
  - **Not reproducible**: the required video, ZIP, CSV, or external comparison
    artifact was not supplied.

## Finding-by-finding disposition

| ID | Verification | Disposition and required follow-up |
| --- | --- | --- |
| F01 | Resolved | Reviewed template matches are authoritative in `pipeline_v2.py`. The residual single-template restriction is real and is covered by T05. |
| F02 | Partly confirmed | `pedagogy_router.py` has 11 modes, but every grammar is still a fixed four-shot form and relies on rule keywords. Generalize only after the benchmark in T01 can measure improvement; covered by T07. |
| F03 | Resolved | The model returns `VisualIntent`/shot-level specifications rather than renderer object graphs. Preserve this boundary. |
| F04 | Resolved with residual | `templates/registry.py` combines BM25, relation, operand, audience, and structural eligibility signals. It does not prove that a template can satisfy the shot's visual/action obligations; covered by T07. |
| F05 | Partly confirmed | The 39-template library and operator compilation are real. Shared card/container structures can still make outputs visually repetitive; covered by T05, T07, and T11. |
| F06 | Partly confirmed | All 42 operators are registered and unsupported operators fail closed. Many use shared generic drawing plugins, so operator-specific pixel semantics remain shallow; covered by T06 and T09. |
| F07 | Resolved | The default semantic catalog loads curated and full indexes, including the 5,910 SVG collection. No inventory-expansion task is justified yet. |
| F08 | Confirmed residual | Unknown concepts can produce a composed mini-diagram, but `semantic_frames.py` explicitly skips a generated asset in a left card slot. Fix in T04. |
| F09 | Resolved with residual | Typed low-level `VisualOperation` values exist, but there is no typed domain-action layer such as transfer, split, merge, consume, or transform. Add it in T06. |
| F10 | Resolved | Layout uses asset dimensions and aspect ratios. The current uncommitted layout work further improves graph/tree geometry; preserve it in T00. |
| F11 | Partly confirmed | Multiple layout strategies exist, but per-shot semantic reflow/composition changes are weak. Covered by T05 and T06. |
| F12 | Partly confirmed | `ShotPlan`, object lifecycle, cleanup, and persistent state are implemented. Template storyboards usually retain one root and cycle through show/dim/highlight; covered by T05 and T06. |
| F13 | Partly confirmed; report overstated | Camera rectangles and target occupancy are geometry-derived, including a desired occupancy band. The missing checks are a minimum meaningful change from the previous camera state and validation of the final rendered focal scale; covered by T09. |
| F14 | Resolved | Camera rendering performs a real crop and resize. FIT/HOLD remaining stable is intentional; crop is not semantic reflow. No separate task. |
| F15 | Confirmed | Motion strategies and adapters exist, but several effects are synthetic masks or fixed polylines rather than transformations derived from semantic state/path geometry. Covered by T06 and T09. |
| F16 | Partly confirmed | Edge boundary capture and an estimated-timing fallback with 0.35 confidence exist. The report's sample-specific timing result is not reproducible. Add explicit provider/fallback tests in T09. |
| F17 | Resolved with risk | TTS parallelism, caching, FFmpeg normalization/concatenation, and a raw-byte fallback exist. Validate the fallback and fail clearly on incompatible audio in T09. |
| F18 | Resolved | The matched-template path avoids a storyboard model call and parallelizes useful stages. An unmatched topic still needs one visual-intent call by design. No standalone task. |
| F19 | Resolved | Provider-level response schemas are enforced by the structured-agent/Gemini client path. Preserve tests. |
| F20 | Resolved with residual | Repairs use JSON Patch. Storyboard repair still replaces a complete `VisualIntent`; make repairs stage-local in T03. |
| F21 | Confirmed | The retry loop reuses some upstream results, but failures after compilation repeat downstream work, and rendered-pixel failures do not receive a local repair. Fix in T03. |
| F22 | Confirmed | `semantic_frames.py` materializes one PNG path per frame, even when frames are linked/copied. This is the primary rendering throughput ceiling; fix in T08. |
| F23 | Confirmed | The layer cache is cleared at beat boundaries, losing reusable static layers. Fix in T08. |
| F24 | Confirmed with scope | Debug artifacts default to enabled and frame traces are emitted per frame when debugging is active. Make production defaults and sampling explicit in T08/T12. |
| F25 | Confirmed | Deterministic quality assigns `diagram_correctness = 1.0` and treats the presence of motion as perfect alignment. Replace these proxies in T02. |
| F26 | Confirmed | Rendered quality checks blankness and broad occupancy, but not clipping, font size, connector crossings, focal visibility, or semantic state change. Add measurable checks in T02. |
| F27 | Confirmed | Multimodal QA is disabled by default and runs only after all frames are rendered. Keep it optional, but make it conditional and earlier for ambiguous/high-risk shots in T02/T03. |
| F28 | Confirmed | `GeminiStoryboardCritic` exists but is not part of the default composite evaluator. Do not simply enable it globally; integrate it behind the risk policy in T02. |
| F29 | Confirmed | There is no 30-topic benchmark suite, golden-frame/perceptual suite, pedagogy-transfer suite, or output-novelty benchmark. Build T01 before making broad architecture changes. |
| F30 | Confirmed | No production novelty fingerprint or recent-output similarity control was found. Add only after correctness metrics in T11. |
| F31 | Partly confirmed; report overstated | The planner is not source-grounded and lacks domain-specific factual validators. Generic relation compatibility, mechanism-chain, dependency, containment, and direction normalization already exist. Extend them in T10 rather than replacing them. The sample graph error itself is not reproducible without its artifact bundle. |

### New-blocker mapping

The report's N01-N08 items mostly repeat F01-F31. They map to this backlog as
follows, so they must not be opened again as duplicate work:

| New blocker | Consolidated task |
| --- | --- |
| N01: single-template/single-operator program | T05 |
| N02: highlights instead of semantic transformations | T06 |
| N03: no obligation/capability suitability check | T07 |
| N04: under-parameterized templates/default-heavy output | T07 |
| N05: generated mini-diagrams skipped in left slot | T04 |
| N06: quality can certify weak output | T02 |
| N07: PNG-per-frame speed ceiling | T08 |
| N08: packaging, eager optional imports, stale docs | T12 |

## Execution order

Do not start the large renderer or visual-program rewrites until T00 and T01
are complete. Otherwise there is no trustworthy baseline for correctness,
quality, or speed.

### P0 — establish a safe baseline and trustworthy gates

#### T00 — checkpoint the current corrective work

- [x] Review the four modified files listed in the baseline and confirm they
  contain only the `visual_state_repeated` and tree/graph layout corrections.
- [x] Keep the educational-quality fix and layout fix in one focused commit,
  or split them into two commits if review shows they are independently
  revertible.
- [x] Run the full test suite and record the result in the commit/PR.

Acceptance criteria:

- 191 tests pass on the checkpointed tree.
- A genuine repeated visual state still fails quality validation.
- Different update payloads are not treated as repeated visual state.
- Tree nodes do not overlap and graph/tree callouts receive a separate lane.

#### T01 — build a reproducible quality and runtime benchmark

Primary area: `tests/benchmark/`, plus a small CLI under `scripts/`.

- [x] Create a versioned set of at least 30 prompts covering process,
  hierarchy, comparison, mechanism, timeline, math, code, biology, finance,
  and deliberately unfamiliar concepts.
- [x] Store prompt, seed/configuration, provider/model identifiers, stage
  timings, selected templates/operators, repair attempts, QA results, and
  artifact paths in a machine-readable result.
- [x] Add deterministic structural assertions and a small reviewed set of
  key/golden frames. Use perceptual thresholds rather than exact PNG equality.
- [x] Measure topic specificity, action coverage, meaningful visual-state
  deltas, composition/camera changes, clipping/readability, audio timing
  confidence, success rate, and wall-clock time.
- [x] Provide an offline/mock tier for CI and an explicitly invoked provider
  tier for release evaluation.

Acceptance criteria:

- The same fixture/configuration can be rerun and compared with a prior result.
- A regression identifies the prompt, beat, metric, and relevant artifact.
- Tests do not silently depend on live providers, mutable external content, or
  a developer's local cache.

#### T02 — replace optimistic quality scores with evidence

Primary files: `app/quality/deterministic.py`, `app/quality/educational.py`,
`app/quality/rendered.py`, `app/quality/composite.py`,
`app/quality/storyboard_critic.py`, and quality models/tests.

- [x] Remove constant `diagram_correctness` and presence-only motion/alignment
  scores.
- [x] Check that each beat's narration/visual obligation is supported by its
  visible objects, relations, state delta, and action—not just IDs or labels.
- [x] Validate relation direction, connector endpoints, recap coverage, and
  transformation preconditions/postconditions.
- [x] On sampled rendered frames, measure safe-area clipping, effective text
  size, text/connector collisions, connector crossings, target visibility,
  focal occupancy, blankness, and density.
- [x] Sample transition frames as well as final frames so motion errors are
  observable.
- [x] Define a risk policy for optional storyboard/multimodal review. Invoke it
  for low-confidence matching, unknown concepts, ambiguous relations, or
  deterministic borderline/failure cases—not indiscriminately for every run.
- [x] Every failing check must emit a stable finding code, object/beat IDs,
  measured versus required values, repair scope, and suggested patch target.

Acceptance criteria:

- No semantic quality dimension receives a perfect score from a constant or
  merely because an object/list exists.
- Unit tests contain known-bad storyboards/frames that fail each new gate and
  known-good counterparts that pass.
- A quality report can explain exactly why a beat failed and which stage owns
  the correction.

#### T03 — make repair and rerender stage-local

Primary file: `app/application/orchestrators/pipeline_v2.py`, with repair
contracts in planning/layout/rendering/quality modules.

- [x] Map every quality finding code to an owning stage and allowed JSON Patch
  paths.
- [x] Preserve successful lesson, pedagogy, narration, TTS, asset, and compiled
  outputs whenever the requested repair does not invalidate them.
- [x] Repair a storyboard intent/beat rather than regenerating the complete
  visual intent when the finding is beat-local.
- [x] Allow layout, motion, camera, and rendered-pixel findings to repair their
  own artifacts without consuming a blind full-pipeline retry.
- [x] Track dependency invalidation explicitly and rerender only affected beats
  and transition windows.
- [x] Detect a repeated identical repair/finding pair and stop with a diagnostic
  explaining why the repair made no progress.

Acceptance criteria:

- Tests prove which stages are and are not reinvoked for each finding class.
- A one-beat clipping failure does not rerun narration/TTS or render unaffected
  beats.
- Rendered-pixel failures receive a bounded local repair attempt before the run
  is rejected.
- Repair exhaustion reports attempts, patches, invalidated stages, and metric
  deltas.

#### T04 — make generated semantic assets reliably visible

Primary files: `app/planning/asset_queries.py`,
`app/rendering/semantic_frames.py`, semantic asset generation/layout tests.

- [x] Stop classifying a composed mini-diagram as a small left-slot icon.
- [x] Give generated diagrams a dedicated object kind/placement contract, or
  produce a card-safe icon variant and a full-diagram variant explicitly.
- [x] Preserve intrinsic aspect ratio and reserve adequate layout space.
- [x] If a generated asset cannot be rendered, fall back visibly and report a
  quality finding; never silently return from the draw path.

Acceptance criteria:

- A forced unknown-concept fixture produces visible, non-clipped artwork.
- The artwork is present in rendered pixels and linked to the intended concept.
- No generated asset is silently skipped because of slot metadata.

### P1 — represent and compile semantic visual actions

#### T05 — introduce a per-shot multi-operator `VisualProgram`

Primary files: `app/domain/visual_intent.py`, `app/domain/strategy.py`,
`app/planning/visual_intent_compiler.py`, and
`app/planning/template_compiler.py`.

- [x] Define a bounded `VisualProgram` with shared objects plus per-shot roots,
  operator instances, state references, and action obligations.
- [x] Remove the `len(operators) == 1` compiler restriction.
- [x] Allow a reviewed template to own one section/shot rather than forcing one
  template root across the whole lesson.
- [x] Define deterministic composition/ownership rules for cross-template
  objects, connectors, IDs, layout regions, and cleanup.
- [x] Keep low-level renderer objects out of model output; the compiler remains
  authoritative.

Acceptance criteria:

- A test lesson compiles at least two semantically justified operators/templates
  across its shots without duplicate IDs or dangling connectors.
- Shared objects persist only when declared; shot-local objects clean up.
- Existing single-template fixtures continue to compile.

#### T06 — add typed semantic actions and operator state machines

Primary files: `app/domain/operations.py`, operator template definitions,
`app/state/engine.py`, `app/motion/planner.py`, and renderer plugins.

- [x] Add a closed, typed semantic-action union above `VisualOperation`, for
  example transfer, route, split, merge, group, compare, consume, produce,
  transform, substitute, accumulate, and trace.
- [x] Define action schemas with operands, relation/direction, preconditions,
  postconditions, duration/easing hints, and reversibility.
- [x] Give each compatible operator an explicit state model and action compiler
  that lowers semantic actions to existing low-level operations.
- [x] Reject unsupported operator/action combinations before rendering.
- [x] Derive trajectories, masks, connector changes, and pixel behavior from
  the action/state delta rather than a strategy name alone.

Acceptance criteria:

- At least 90% of transformation beats in the benchmark contain a supported
  semantic action or fail with a precise unsupported-action finding.
- Every transformation beat has a non-trivial, testable semantic state delta.
- Tests cover valid lowering, invalid operands, missing preconditions, cleanup,
  and reversible transitions.

#### T07 — match templates by capability and extract real parameters

Primary files: `app/templates/registry.py`,
`app/planning/template_compiler.py`, template metadata, and pedagogy routing.

- [x] Add machine-readable template capabilities: supported relation types,
  actions, operand cardinality, pedagogy roles, and layout constraints.
- [x] Make capability/obligation compatibility a hard eligibility check before
  lexical ranking.
- [x] Replace the small global safe-key copy with reviewed per-template
  parameter schemas/extractors.
- [x] Record extraction provenance and distinguish extracted values from
  defaults.
- [x] Permit shot counts to follow the lesson/pedagogy requirements rather than
  always forcing four shots, while keeping explicit upper/lower bounds.

Acceptance criteria:

- A high-BM25 but capability-incompatible template cannot win selection.
- Required parameters cannot silently fall back to generic defaults.
- Benchmark output reports match confidence, capability evidence, parameter
  provenance, and default usage.

#### T10 — add factual grounding and domain validation

Primary files: lesson planning/validation models and provider prompts; extend
the existing generic relation validators.

- [x] Add optional source/reference inputs and retain claim-level provenance in
  the lesson plan.
- [x] Validate generated claims and relation directions against supplied
  sources when sources are present.
- [x] Add a plugin-style domain-validator interface; start only with domains
  represented in the benchmark and with clear deterministic invariants.
- [x] Route unsupported or low-confidence claims to a review/failure state
  instead of inventing a relation.
- [x] Preserve the existing compatibility, mechanism-chain, dependency,
  containment, and normalization checks as the generic layer.

Acceptance criteria:

- A deliberately reversed or unsupported domain relation fails with evidence.
- Every source-grounded factual claim can be traced to its source identifier.
- Domain validators are optional and do not make the base package import their
  heavy dependencies eagerly.

### P2 — remove the rendering throughput ceiling

#### T08 — render keyframes/layers and stream video output

Primary files: `app/rendering/semantic_frames.py`, video composition code,
debug recorder/settings, and renderer performance tests.

- [ ] Split static background, persistent object, changed object, connector,
  overlay, and camera-composite layers.
- [ ] Keep valid static/persistent layers across beat boundaries and invalidate
  them by content/style/layout signature.
- [ ] Render semantic keyframes plus interpolated deltas rather than rebuilding
  and writing every full frame.
- [ ] Stream raw frames to FFmpeg, or use a measured equivalent, without a PNG
  file per output frame.
- [ ] Preserve a bounded diagnostic frame sample on failure.
- [ ] Default debug artifacts off for production and sample frame traces by
  event/keyframe; retain an explicit full-trace mode.
- [ ] Add partial beat/window rerender support required by T03.

Acceptance criteria:

- Pixel/perceptual comparison shows no material regression on the golden set.
- A successful production run does not create one PNG and one trace record per
  output frame.
- The benchmark records stage time, peak memory, cache hit rate, keyframe count,
  and encoded-frame throughput.
- Two-minute benchmark targets, measured on a documented machine/profile, are
  median <= 180 seconds and p90 <= 240 seconds, or an explicitly approved
  replacement target based on the T01 baseline.

### P3 — improve camera, motion, and audio fidelity

#### T09 — make movement and synchronization evidence-based

Primary files: `app/camera/planner.py`, `app/motion/planner.py`, reveal/motion
adapters, `app/core/audio_manager.py`, and `app/audio/alignment.py`.

- [ ] Require a configurable minimum camera crop/center/scale delta when the
  pedagogy calls for a camera change; validate the final rendered target scale.
- [ ] Make FIT/HOLD exemptions explicit rather than treating them as failed
  motion.
- [ ] Use connector/path geometry and semantic action deltas for travel,
  transfer, routing, morphing, and accumulation animations.
- [ ] Test real Edge-TTS boundary parsing with a recorded provider fixture.
- [ ] Report provider, estimated, and mixed timing coverage/confidence; add a
  forced-alignment adapter only if benchmark error justifies it.
- [ ] Validate concatenated audio formats and fail clearly instead of accepting
  incompatible raw-byte concatenation.

Acceptance criteria:

- Camera-change beats exceed the configured visible-change threshold while
  retaining all required target edges.
- Path/action tests assert positions at multiple intermediate timestamps.
- At least 80% of narrated words have provider or validated aligned timing in
  the release benchmark, with no silent low-confidence downgrade.

### P4 — breadth, novelty, packaging, and release discipline

#### T11 — add controlled novelty after correctness

Primary areas: template/operator selection, layout variants, style variants,
and benchmark history.

- [ ] Define a structural fingerprint from operator sequence, template family,
  layout topology, camera plan, action sequence, and palette/style tokens.
- [ ] Compare candidates with recent successful outputs and select a different
  compatible variant when similarity exceeds a measured threshold.
- [ ] Add reviewed layout/operator variants; never sacrifice obligation or
  factual correctness merely to be different.
- [ ] Track novelty as a benchmark metric, not an unconditional quality gate,
  until false-positive behavior is understood.

Acceptance criteria:

- Repeated benchmark runs expose their fingerprints and similarity scores.
- Novelty selection cannot choose an incompatible template/action pair.
- Golden correctness and readability gates remain unchanged or improve.

#### T12 — clean package boundaries and stale documentation

Primary files: package `__init__.py` modules, dependency metadata, settings,
`PROJECT_HANDOFF.md`, and setup/run documentation.

- [ ] Replace eager imports of optional audio/provider/multimodal dependencies
  with lazy adapters or explicit extras.
- [ ] Document minimal, audio, provider, multimodal, and development install
  profiles and their failure messages.
- [ ] Change production-oriented debug defaults and document how to enable full
  artifacts.
- [ ] Update stale handoff claims that templates are not instantiated and that
  all camera behavior is FIT-only.
- [ ] Add a clean-environment import/startup test for the minimal package.

Acceptance criteria:

- Core domain/planning modules import without optional TTS or provider SDKs.
- Missing extras produce a targeted instruction rather than an import-time
  crash.
- Documentation matches the current template, camera, quality, and renderer
  behavior.

#### T13 — enforce release gates in CI/release reports

Depends on T01, T02, T06, T08, T09, and T10.

- [ ] Publish benchmark deltas and artifacts for release candidates.
- [ ] Gate on reliability, clipping/readability, factual/relation validity,
  action coverage, state deltas, and approved runtime targets.
- [ ] Start non-deterministic/model-judged metrics as reported signals and only
  promote them to gates after repeatability is measured.
- [ ] Require an explicit waiver, owner, and expiry for any failed release gate.

Initial targets to validate and tune from the T01 baseline:

- Reliability: at least 95% successful benchmark runs.
- Action coverage: at least 90% of transformation beats.
- Topic specificity: at least 75% of primary visible objects tied to lesson
  concepts/actions, using a documented metric.
- Readability: no safe-area clipping; effective body text at least 24 px at
  1080p unless an accessibility-approved alternative exists.
- Semantics: no unsupported/reversed required relation; every transformation
  beat has a state delta; recap covers declared learning objectives.
- Timing: at least 80% validated/provider word alignment coverage.
- Performance: use the T08 documented two-minute target.

## Explicit non-tasks

- Do not add more SVGs merely to increase the asset count; retrieval,
  suitability, visibility, and semantic actions are the current constraints.
- Do not expose raw renderer primitives to the model. Keep model output bounded
  and compile it deterministically.
- Do not globally enable an expensive LLM/vision critic before deterministic
  checks and a risk policy exist.
- Do not adopt Graphviz, SymPy, RDKit, or another domain engine solely because
  the audit names it. Add one behind T10 only when a benchmark domain and a
  concrete invariant justify the dependency.
- Do not use the unsupplied sample video or Simi comparison as a release gate.
  Reproduce any comparison from versioned inputs and documented settings first.

## Recommended delivery slices

1. **Safety slice:** T00, T01, T02, T03, T04.
2. **Semantic-program slice:** T05, T06, T07, T10.
3. **Performance slice:** T08.
4. **Fidelity slice:** T09.
5. **Productization slice:** T11, T12, T13.

Each slice should land behind tests and benchmark evidence. Re-estimate later
slices after the preceding slice has produced measured results; the calendar
estimates in the audit are opinions, not evidence from this repository.
