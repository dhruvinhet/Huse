"""Clean-environment tests for lazy optional dependency boundaries."""

import os
import subprocess
import sys


_BLOCKER = r'''
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    blocked = {"edge_tts", "google", "cv2", "supabase"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockOptional())
'''


def _run(code: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + code],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, **environment},
    )


def test_minimal_domain_planning_and_quality_import_without_optional_sdks() -> None:
    """Core semantic packages must not load TTS, provider, or video SDKs."""

    result = _run(r'''
import app.domain
import app.planning
import app.quality
assert "edge_tts" not in sys.modules
assert "google" not in sys.modules
assert "cv2" not in sys.modules
print("minimal-import-ok")
''')

    assert result.returncode == 0, result.stderr
    assert "minimal-import-ok" in result.stdout


def test_missing_audio_profile_has_targeted_install_instruction() -> None:
    """Using TTS without its profile names the exact requirements file."""

    result = _run(r'''
from app.core.audio_manager import AudioManager
try:
    AudioManager().generate(object())
except ImportError as exc:
    print(str(exc))
else:
    raise AssertionError("missing audio dependencies were accepted")
''')

    assert result.returncode == 0, result.stderr
    assert "requirements-audio.txt" in result.stdout


def test_missing_gemini_profile_has_targeted_install_instruction() -> None:
    """Using Gemini without its SDK names the provider install profile."""

    result = _run(
        r'''
from app.services.gemini_client import GeminiClient
try:
    GeminiClient()
except ImportError as exc:
    print(str(exc))
else:
    raise AssertionError("missing provider dependencies were accepted")
''',
        AI_PROVIDER="gemini",
        GEMINI_API_KEY="test-key",
    )

    assert result.returncode == 0, result.stderr
    assert "requirements-provider.txt" in result.stdout
