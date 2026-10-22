"""A fresh user can exercise the pipeline without downloading foundation weights."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


def test_offline_example_from_another_directory(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    script = Path(__file__).resolve().parents[1] / "examples/offline_smoke.py"
    environment = dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable,str(script)],cwd=tmp_path,
        capture_output=True,text=True,timeout=45,env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "passed"' in result.stdout
    assert '"probability_not_claimed": true' in result.stdout
    assert "no accuracy measurement" in result.stdout
