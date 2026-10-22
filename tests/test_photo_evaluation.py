import json
import numpy as np
from PIL import Image
import pytest

from pmsa.photo_evaluation import inventory_photos, summarize_real_scores


def test_inventory_exact_duplicates_and_no_pixel_export(tmp_path):
    image = Image.new("RGB", (17, 13), (30, 80, 120))
    image.save(tmp_path / "a.png")
    image.save(tmp_path / "b.png")
    Image.new("RGB", (19, 14), (120, 20, 50)).save(tmp_path / "c.png")
    inventory = inventory_photos(tmp_path)
    assert inventory["file_count"] == 3
    assert inventory["unique_decoded_count"] == 2
    assert inventory["exact_decoded_duplicates"] == 1
    assert inventory["representative_indices"] == [0, 2]
    assert not inventory["independence_verified"]
    assert "pixels" not in json.dumps(inventory)


def test_no_photo_links_escape_scope(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (5, 5)).save(tmp_path / "unrelated.png")
    (photos / "escape.png").symlink_to(tmp_path / "unrelated.png")
    with pytest.raises(ValueError, match="escapes"):
        inventory_photos(photos)


def test_empty_and_unreadable_photos_fail(tmp_path):
    with pytest.raises(ValueError, match="No supported"):
        inventory_photos(tmp_path)
    (tmp_path / "corrupt.jpg").write_bytes(b"not an image")
    with pytest.raises(Exception):
        inventory_photos(tmp_path)


def test_input_failure_is_not_a_correct_real_prediction(tmp_path):
    Image.new("RGB", (5, 5)).save(tmp_path / "readable.png")
    (tmp_path / "corrupt.jpg").write_bytes(b"invalid")
    inventory = inventory_photos(tmp_path, record_decode_failures=True)
    assert inventory["file_count"] == 2
    assert inventory["decoded_file_count"] == 1
    assert inventory["decode_failures"] == 1
    assert inventory["unique_decoded_count"] == 1
    assert inventory["exact_decoded_duplicates"] == 0
    assert inventory["representative_indices"] == [1]


def test_real_only_operating_count_keeps_abstentions_in_denominator():
    report = summarize_real_scores([-1., 0., .1, .9], 0.)
    assert report["false_alarms"] == 2  # strict >, equality is negative
    assert report["fpr"]["point"] == .5
    assert report["n_real"] == 4
    assert report["ui_uncertain"] == 2
    assert report["ui_ai_signal"] == 1
    assert report["ui_no_ai_signal"] == 1
    assert "auc" not in report and "tpr" not in report


def test_zero_errors_not_certified_zero_population_fpr():
    report = summarize_real_scores([-1.] * 37, 0.)
    assert report["fpr"]["point"] == 0
    assert .09 < report["fpr"]["hi"] < .10


@pytest.mark.parametrize("scores,tau,band", [([], 0., .25), ([np.nan], 0., .25),
    ([0.], np.inf, .25), ([[0.]], 0., .25), ([0.], 0., -1.)])
def test_bad_operating_inputs_fail(scores, tau, band):
    with pytest.raises(ValueError):
        summarize_real_scores(scores, tau, band)
