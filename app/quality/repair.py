"""Map quality findings to bounded stage-local repair plans."""

from app.domain.quality import QualityReport
from app.domain.repair import RepairPlan, RepairStage


class QualityRepairPlanner:
    """Assign one owner stage and invalidate only its dependants."""

    _ORDER = [
        RepairStage.STORYBOARD,
        RepairStage.NARRATION,
        RepairStage.ASSETS,
        RepairStage.AUDIO,
        RepairStage.STATE,
        RepairStage.LAYOUT,
        RepairStage.MOTION,
        RepairStage.CAMERA,
        RepairStage.RENDERER,
        RepairStage.COMPOSER,
    ]
    _DOWNSTREAM = {
        RepairStage.STORYBOARD: set(_ORDER),
        RepairStage.NARRATION: {
            RepairStage.NARRATION,
            RepairStage.AUDIO,
            RepairStage.MOTION,
            RepairStage.CAMERA,
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.ASSETS: {
            RepairStage.ASSETS,
            RepairStage.LAYOUT,
            RepairStage.MOTION,
            RepairStage.CAMERA,
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.AUDIO: {
            RepairStage.AUDIO,
            RepairStage.MOTION,
            RepairStage.CAMERA,
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.STATE: {
            RepairStage.STATE,
            RepairStage.LAYOUT,
            RepairStage.MOTION,
            RepairStage.CAMERA,
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.LAYOUT: {
            RepairStage.LAYOUT,
            RepairStage.MOTION,
            RepairStage.CAMERA,
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.MOTION: {
            RepairStage.MOTION,
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.CAMERA: {
            RepairStage.CAMERA,
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.RENDERER: {
            RepairStage.RENDERER,
            RepairStage.COMPOSER,
        },
        RepairStage.COMPOSER: {RepairStage.COMPOSER},
    }
    _TARGETS = {
        "storyboard": RepairStage.STORYBOARD,
        "narration": RepairStage.NARRATION,
        "assets": RepairStage.ASSETS,
        "audio": RepairStage.AUDIO,
        "state": RepairStage.STATE,
        "layout": RepairStage.LAYOUT,
        "motion": RepairStage.MOTION,
        "camera": RepairStage.CAMERA,
        "renderer": RepairStage.RENDERER,
        "composer": RepairStage.COMPOSER,
    }
    _CODE_OWNERS = {
        "rendered_opening_blank": RepairStage.MOTION,
        "rendered_samples_blank": RepairStage.RENDERER,
        "rendered_canvas_underused": RepairStage.LAYOUT,
        "rendered_canvas_overfilled": RepairStage.LAYOUT,
        "rendered_safe_area_clipped": RepairStage.LAYOUT,
        "rendered_effective_font_too_small": RepairStage.LAYOUT,
        "rendered_focal_target_missing": RepairStage.CAMERA,
        "narration_visual_alignment_invalid": RepairStage.MOTION,
    }
    _PATCH_PREFIXES = {
        RepairStage.STORYBOARD: ("/beats", "/initial_objects", "/final_learning_summary"),
        RepairStage.NARRATION: ("/phrases",),
        RepairStage.ASSETS: ("/assets",),
        RepairStage.AUDIO: ("/audio", "/phrases"),
        RepairStage.STATE: ("/states",),
        RepairStage.LAYOUT: ("/layout",),
        RepairStage.MOTION: ("/events", "/motion", "/cues"),
        RepairStage.CAMERA: ("/camera", "/cues"),
        RepairStage.RENDERER: ("/render", "/frames"),
        RepairStage.COMPOSER: ("/composer",),
    }

    def plan(self, report: QualityReport) -> RepairPlan:
        """Choose the earliest owning stage across all actionable findings."""

        if not report.findings:
            raise ValueError("cannot plan repair without quality findings")
        owners = [self._owner(item.code, item.repair_target) for item in report.findings]
        owner = min(owners, key=self._ORDER.index)
        relevant = [
            finding
            for finding, finding_owner in zip(report.findings, owners, strict=True)
            if finding_owner is owner
        ]
        prefixes = self._PATCH_PREFIXES[owner]
        requested_paths = [
            path
            for finding in relevant
            for path in finding.patch_paths
            if path.startswith(prefixes)
        ]
        allowed_paths = list(dict.fromkeys(requested_paths or [prefixes[0]]))
        invalidated = [
            stage for stage in self._ORDER if stage in self._DOWNSTREAM[owner]
        ]
        return RepairPlan(
            owner_stage=owner,
            invalidated_stages=invalidated,
            finding_codes=sorted({finding.code for finding in relevant}),
            findings=relevant,
            beat_ids=sorted({
                finding.beat_id for finding in relevant if finding.beat_id
            }),
            object_ids=sorted({
                object_id for finding in relevant for object_id in finding.object_ids
            }),
            frame_numbers=sorted({
                number for finding in relevant for number in finding.frame_numbers
            }),
            allowed_patch_paths=allowed_paths,
            fingerprint=RepairPlan.fingerprint_for(owner, relevant),
        )

    def _owner(self, code: str, repair_target: str | None) -> RepairStage:
        if code in self._CODE_OWNERS:
            return self._CODE_OWNERS[code]
        return self._TARGETS.get(repair_target or "", RepairStage.STORYBOARD)
