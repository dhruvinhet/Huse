# Whiteboard AI Video Generator

## Overview

This Python 3.12 project generates narrated educational whiteboard videos. It now contains two versioned pipelines:

- **V1** is the stable script-to-primitives pipeline with Edge-TTS, deterministic Pillow/SVG frame rendering, phrase-safe audio timing, FFmpeg composition, verified output streams, and complete debug bundles.
- **V2** is the semantic storyboard-first architecture. It separates lesson planning, concept graphs, visual storyboards, narration, persistent visual state, constraint layout, semantic animation, attention, camera planning, assets, quality evaluation, rendering, and composition.

V2 is the default pipeline. V1 remains available as an explicit compatibility mode. See [V2 architecture](docs/architecture-v2.md).

New contributors should begin with [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md). It covers project history, architecture, setup, debugging, security, current limitations, and the recommended next milestone.

## V2 pipeline

```text
Request
  -> Lesson Plan and Concept Graph
  -> Visual Strategies and Educational Templates
  -> Persistent Storyboard
  -> Storyboard-Driven Narration
  -> Edge-TTS and Phrase Alignment
  -> Semantic Asset Resolution
  -> Immutable Visual States
  -> Hierarchical Constraint Layout
  -> Semantic Motion, Attention, and Camera
  -> Deterministic and Optional Multimodal Quality Gates
  -> Semantic PNG Frames
  -> Existing Verified FFmpeg Composer
```

The renderer never infers educational meaning, and AI planners never emit pixel coordinates.

### V2 visual-quality capabilities

- Shared whiteboard design tokens for typography, semantic color, strokes, cards, and emphasis.
- Purpose-driven teaching beats: introduce, demonstrate, compare, transform, connect, emphasize, and summarize.
- General educational templates for comparisons, timelines, cycles, cause/effect, architecture, flowcharts, funnels, Venn diagrams, charts, equation derivations, and code traces.
- Editable concept-specific SVG pictograms for people, teams, data, cloud, servers, devices, security, finance, networks, AI, education, and health.
- Path-progress connector drawing, handwriting/stroke reveals, marker-tip feedback, semantic fades, growth, and erasure.
- Edge-TTS word-boundary capture and word-onset animation scheduling, with phrase timing retained as a fallback.
- Geometry-aware fit, focus, zoom, pan, and tracking camera choreography based on beat purpose.
- Automatic preflight checks for clipping, unreadable geometry, sibling overlap, asset readiness, and excessive density.
- Educational checks for concept coverage, prerequisite order, worked examples, teaching progression, final summaries, and audience-level narration density.

## Important directories

```text
app/
├── agents/             # bounded structured AI planners and critics
├── application/        # ports, orchestrators, and public use cases
├── audio/              # storyboard narration adapter and phrase alignment
├── camera/             # virtual camera planning
├── core/               # preserved V1 pipeline
├── domain/             # versioned V2 contracts and schema registry
├── knowledge/          # built-in and reviewed persistent strategies
├── layout/             # deterministic hierarchical layout
├── motion/             # semantic animation scheduling
├── observability/      # content-addressed checkpoints and provenance
├── planning/           # attention and density policies
├── quality/            # deterministic, AI, and multimodal evaluation
├── rendering/          # V2 semantic renderer and composer adapter
├── renderers/          # preserved Pillow/SVG renderers
├── semantic_assets/    # catalog and editable line-art fallback
├── state/              # persistent visual state transitions
├── templates/          # semantic and topic-specific diagram templates
└── timeline/           # phrase-aligned manifest generation
```

## Requirements

- Python 3.12
- FFmpeg available on `PATH`, or configured with `FFMPEG_PATH`
- Gemini or NVIDIA API key for real AI planning runs

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env`:

```dotenv
AI_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash
NVIDIA_API_KEY=
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
NVIDIA_VISION_MODEL=google/gemma-3n-e4b-it
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MAX_TOKENS=16384
OUTPUT_DIR=outputs
TEMP_DIR=temp
LOG_LEVEL=INFO
FFMPEG_PATH=
DEBUG_ARTIFACTS=true
DEBUG_DIR=outputs/debug
PIPELINE_VERSION=v2
V2_MAX_REPAIR_ATTEMPTS=2
V2_ENABLE_MULTIMODAL=false
```

Set `AI_PROVIDER=gemini` to use Gemini, or `AI_PROVIDER=nvidia` (also accepted:
`nvidea`) to use NVIDIA's hosted OpenAI-compatible API. When NVIDIA is selected,
set `NVIDIA_API_KEY`; `NVIDIA_MODEL` controls text generation and
`NVIDIA_VISION_MODEL` controls optional multimodal quality checks.

Use `PIPELINE_VERSION=v2` to run the semantic pipeline, or set it to `v1` only when compatibility with the primitive renderer is required. Multimodal verification is optional because it adds model calls, latency, and cost.

## Run

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

The prompt asks for a topic and writes `outputs/final_video.mp4`.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests do not make real Gemini, TTS, or FFmpeg network/process calls unless explicitly configured as an external integration run.

## JSON Schemas

V2 contracts are registered in `app/domain/schemas.py`. Regenerate committed schemas with:

```powershell
.\.venv\Scripts\python.exe export_schemas.py
```

Generated files are written to `docs/schemas/`.

## Debugging

With `DEBUG_ARTIFACTS=true`, each real run records inputs, validated AI artifacts, narration and alignment, assets, visual states, layout, motion, camera, quality reports, frame traces, FFmpeg commands, stream verification, errors, and stage timings under `outputs/debug/runs/`.

Inspect the latest V1 debug report:

```powershell
.\.venv\Scripts\python.exe debug_report.py
```

API keys are never recorded. Run status is atomically finalized in both `run.json` and `result.json`.

## Compatibility

- Existing V1 models and public methods remain available.
- `VideoGenerationService` selects V1 or V2 behind a stable use-case facade.
- Explicit legacy adapters convert V1 scripts to V2 storyboards and supported V2 states to V1 render scenes.
- Unsupported V2 objects fail validation or use an explicit semantic fallback; they do not silently become empty image placeholders.
