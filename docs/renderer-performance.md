# Renderer performance profile

Production rendering streams raw RGB frames directly to FFmpeg and retains only
the first, middle, and last frame of each scene for diagnostics. Set
`OutputProfile.keep_frames=true` only for golden-image development. Debug
artifacts default to off; `DEBUG_ARTIFACTS=true` enables event/keyframe traces,
and `DEBUG_FRAME_TRACE_FULL=true` explicitly enables per-frame traces.

Run the reproducible two-minute renderer/encoder profile with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_renderer.py
```

The report at `outputs/benchmark/t08-performance/report.json` records the OS,
Python and FFmpeg paths, resolution, duration, per-run stage time, Python peak
memory, cache-hit rate, keyframe count, encoded throughput, median, and p90.
This is the approved local replacement baseline for the T01 offline suite,
which intentionally excludes provider, TTS, asset-network, and full-HD costs.
Release reports must label it as renderer-only and must not present it as full
end-to-end provider latency.
