"""Prompt template for structured educational script generation."""

import json


def build_script_prompt(topic: str) -> str:
    """Build the Gemini prompt for a concise educational video script."""

    serialized_topic = json.dumps(topic.strip(), ensure_ascii=False)
    introduction = (
        "Create a clear, accurate, and engaging educational script about the "
        f"topic {serialized_topic}."
    )
    content_guidance = (
        "The complete script should be suitable for an approximately 60-second "
        "whiteboard video. Divide the explanation into logically ordered, "
        "concise scenes. Narration should be natural when spoken aloud, and "
        "every visual instruction should directly support the narration. "
        "Use only visuals that the local whiteboard renderer supports: title, "
        "text, arrow, box, and circle. Never request images, illustrations, "
        "photos, generated artwork, or icons. Use at most three visuals per "
        "scene. Give each visual a different position chosen from title, left, "
        "right, top, bottom, and center so visual elements never overlap. For "
        "arrow, box, and circle visuals, set content to the matching primitive "
        "name. Keep displayed text short and readable."
    )
    output_rules = (
        "Return ONLY one valid JSON object. Do not include markdown, "
        "explanations, commentary, or code fences. Do not place any text before "
        "or after the JSON object."
    )
    schema = f"""Use exactly this JSON structure and field names:
{{
  "title": "...",
  "topic": {serialized_topic},
  "total_duration": 60,
  "scenes": [
    {{
      "scene_number": 1,
      "title": "...",
      "narration": "...",
      "estimated_duration": 12,
      "visuals": [
        {{
          "type": "text",
          "content": "...",
          "position": "center",
          "animation": "write"
        }}
      ]
    }}
  ]
}}"""
    validation_rules = (
        "All required fields must be present. Use positive durations, number "
        "scenes sequentially starting at 1, and ensure the scene durations are "
        "appropriate for the requested total duration. Every scene must include "
        "at least one supported visual instruction."
    )

    return "\n\n".join(
        (
            introduction,
            content_guidance,
            output_rules,
            schema,
            validation_rules,
        )
    )
