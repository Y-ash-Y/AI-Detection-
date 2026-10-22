"""Disposable synthetic image-to-verdict smoke; NOT a usable detector model.

No SigLIP download, benchmark images, personal images or production artifacts.
All pictures are generated noise; labels are arbitrary test classes, not origin.
"""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run():
    import numpy as np
    import torch
    from PIL import Image
    from pmsa.artifacts import save_bundle, sha256_file
    from pmsa.calibration import Calibrator
    from pmsa.data.manifest import ImageRecord, Manifest
    from pmsa.features.extract import extract_backbone
    from pmsa.inference import Detector
    from pmsa.models import FusionDetector, StreamSpec

    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    with TemporaryDirectory(prefix="pmsa-synthetic-smoke-") as temporary:
        root = Path(temporary)
        records = []
        for i in range(128):
            pixels = rng.integers(0, 256, (32,40,3), dtype=np.uint8)
            path = root / f"fixture_{i:03d}.png"
            Image.fromarray(pixels).save(path)
            records.append(ImageRecord(str(path), i % 2, "synthetic_fixture", "test_class", f"fixture-{i}"))
        fs = extract_backbone(Manifest(records), "forensic", root / "features.npz",
            device="cpu", batch_size=16, num_workers=0, image_size=32, normalize_jpeg=True)
        model = FusionDetector([StreamSpec("forensic",66,8)], seed=0)
        model.fit([fs.features[:64]],fs.labels[:64], [fs.features[64:80]],fs.labels[64:80],
                  epochs=1,batch_size=16,verbose=False)
        # Four disjoint fixture roles; only class-zero calibration rows set tau.
        calibration = np.arange(80,120)[fs.labels[80:120] == 0]
        cal = Calibrator.fit(model.score([fs.features[calibration]]),.05,real_domain="synthetic_fixture_only")
        folder = root / "bundle"
        folder.mkdir()
        model.save(folder / "fusion.pt")
        cal.save(folder / "calibrator.json")
        partitions = {"train":list(range(64)),"validation":list(range(64,80)),
                      "calibration":list(range(80,120)),"test":list(range(120,128))}
        (folder / "splits.json").write_text(json.dumps({"synthetic_only":True,"partitions":partitions}))
        save_bundle(folder,"fusion.pt","fusion",{"forensic":fs.metadata})
        detector = Detector(folder / "fusion.pt",folder / "calibrator.json",device="cpu")
        target = Path(fs.paths[120])
        original_hash = sha256_file(target)
        with Image.open(target) as image:
            features = detector._features(image)
            verdict = detector.predict(image,image_path=target)
        checks = {
            "cached_vs_serving_features":bool(np.allclose(features[0],fs.features[120:121],atol=1e-6,rtol=1e-6)),
            "saved_vs_serving_score":bool(np.isclose(verdict.score,model.score([fs.features[120:121]])[0],atol=1e-6)),
            "signed_contributions_plus_bias":bool(np.isclose(sum(verdict.per_stream.values())+verdict.combiner_bias,verdict.score,atol=1e-6)),
            "strict_threshold_decision":verdict.calibrated_fake == bool(verdict.score > cal.tau),
            "probability_not_claimed":verdict.ai_probability is None,
            "source_unchanged":original_hash == sha256_file(target),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Synthetic pipeline contract failed: {checks}")
    return {"status":"passed","checks":checks,
            "scope":"Synthetic forensic-only software smoke; no accuracy measurement or deployable model. Temporary files removed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(),indent=2))


if __name__ == "__main__":
    main()
