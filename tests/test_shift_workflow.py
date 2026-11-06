import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pmsa.calibration import Calibrator
from pmsa.eval.shift_matrix import evaluate_cell, run_matrix
from pmsa.features import FeatureSet


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scoring_uses_checkpoint_order_and_refuses_reversed_rows(monkeypatch, tmp_path):
    cli = load_script("04_score_and_calibrate.py")
    for name, value in (("siglip", 1), ("forensic", 10)):
        fs = FeatureSet(np.array([[value], [value * 2]]), np.array([0, 1]),
                        np.array(["a", "b"]), np.array(["x", "x"]),
                        np.array(["real", "g"]), name)
        fs.save(tmp_path / f"{name}_test.npz")
    seen = []

    def score(xs):
        seen.append(tuple(int(x[0, 0]) for x in xs))
        return np.array([0., 1.])

    model = SimpleNamespace(specs=[SimpleNamespace(name=n) for n in ("forensic", "siglip")],
                            score=score)
    monkeypatch.setattr(cli.FusionDetector, "load", lambda *a, **kw: model)
    monkeypatch.setattr(cli, "get_device", lambda: "cpu")
    base, _ = cli.score_cache(tmp_path, "test", "mock.pt", "fusion", "siglip")
    assert base.backbone == "forensic"
    assert seen == [(10, 1)]
    path = tmp_path / "forensic_test.npz"
    FeatureSet.load(path).select(np.array([1, 0])).save(path)
    with pytest.raises(ValueError, match="not aligned"):
        cli.score_cache(tmp_path, "test", "mock.pt", "fusion", "siglip")
    assert seen == [(10, 1)]


def test_shift_cli_reuses_base_calibration_and_preserves_reports(monkeypatch, tmp_path):
    cli = load_script("05_shift_experiment.py")
    # One duplicate real per domain must not become a separate test draw.
    domains = np.repeat(["base", "shift"], 43)
    scores = np.tile(np.r_[np.arange(40), 0., 22., 40.], 2)
    scores[domains == "shift"] += 5
    labels = np.tile(np.r_[np.zeros(41), 1, 1], 2)
    sources = np.where(labels == 0, "real", "g")
    groups = np.concatenate([np.array([f"{d}-{i}" for i in [*range(40), 0, 41, 42]])
                             for d in ("base", "shift")])
    archive = tmp_path / "scores.npz"
    report = tmp_path / "report.json"
    np.savez(archive, score=scores, label=labels, domain=domains, source=sources, groups=groups)
    config = SimpleNamespace(calibration=SimpleNamespace(alpha=.05, method="conformal",
                                                         cal_fraction=.5, seed=0))
    monkeypatch.setattr(cli.Config, "load", lambda p: config)
    monkeypatch.setattr(sys, "argv", ["05", "--scores", str(archive), "--base-domain", "base",
                                      "--base-generator", "g", "--out", str(report)])
    cli.main()
    data = json.loads(report.read_text())
    for d, split in data["real_partitions"].items():
        ca, te = map(set, (split["calibration_groups"], split["test_groups"]))
        assert len(ca) == len(te) == 20 and ca.isdisjoint(te)
        assert ca | te == {f"{d}-{i}" for i in range(40)}
        cell = next(c for c in data["cells"] if c["real_domain"] == d)
        assert cell["n_real"] == 20
        expected_tau = max(scores[np.isin(groups, list(ca))])
        assert cell["tau_recalibrated"] == expected_tau
        if d == "base":
            assert cell["tau_original"] == expected_tau
    with pytest.raises(FileExistsError):
        cli.main()


def test_fpr_reduction_is_not_target_attainment():
    base = Calibrator.fit(np.full(100, -1.), .05)
    cell = evaluate_cell("D", base, np.r_[np.zeros(60), np.ones(40)], [-.5, .5],
                         np.zeros(100), "shift", "g", .05)
    data = run_matrix([cell])
    assert cell.fpr_original_tau == 1 and cell.fpr_recalibrated == .4
    assert cell.tpr == 1 and cell.tpr_recalibrated == .5
    assert not data["summary"]["all_domains_observed_recal_fpr_at_or_below_target"]
    assert "guarantee_restored" not in data["summary"]
