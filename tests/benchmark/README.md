# Huse benchmark fixtures

`cases.v1.json` is the stable 30-topic suite. The default CLI tier is offline:
it runs production deterministic planning, state, layout, motion, camera, and
quality components without provider calls or media I/O.

Run it with:

```powershell
python scripts/run_benchmark.py --mode offline --output outputs/benchmark/offline
```

Live provider runs are explicit and must record provider/model identity:

```powershell
python scripts/run_benchmark.py --mode provider --provider-id gemini --model-id MODEL --output outputs/benchmark/provider
```

Use `--baseline path/to/report.json` to write a machine-readable comparison.
Golden images use perceptual distance thresholds, not byte equality. Add a
frame only after it has been reviewed, document its purpose in `index.json`,
and keep the threshold narrow enough to detect a meaningful visual regression.
