import cv2
import numpy as np
import pytest

from gui_segesferoides import postprocess
from gui_segesferoides.postprocess_store import PostprocessStore

STATS = {"1": {"diameter": {"mean": 20.0, "p5": 10.0, "p95": 30.0},
               "circularity": {"mean": 0.9, "p5": 0.8, "p95": 0.95}}}


def disc(mask, x, y, r, c):
    cv2.circle(mask, (x, y), r, c, -1)


def test_default_rules_delete_small_and_reclassify_big_irregular():
    mask = np.zeros((200, 200), np.uint8)
    disc(mask, 20, 20, 3, 1)                       # diámetro ~7 < p5 (10): se descarta
    disc(mask, 60, 60, 12, 1)                      # diámetro ~25: queda
    mask[120:190, 120:135] = 1                     # largo y flaco: 70x15, diámetro ~37 > p95, circ. < p5 -> clase 2
    reasons, warnings = [], []
    out, dropped = postprocess.apply(mask, postprocess.defaults(), STATS, reasons, warnings)
    assert out[20, 20] == 0 and dropped[20, 20] == 1 and out[60, 60] == 1 and out[150, 125] == 2
    assert [r["rule"] for r in reasons] == ["Descartar menores al p5 de la clase 1"]
    assert reasons[0]["conditions"][0]["reference"] == "c1.p5" and reasons[0]["conditions"][0]["value"] == 10.0
    assert warnings == []


def test_rule_without_stats_is_skipped_and_reported():
    mask = np.zeros((50, 50), np.uint8)
    disc(mask, 20, 20, 3, 1)
    warnings = []
    out, dropped = postprocess.apply(mask, postprocess.defaults(), {}, warnings=warnings)
    assert out[20, 20] == 1 and not dropped.any() and len(warnings) == 2


def test_reclass_chains_into_following_rules_and_keep_protects():
    mask = np.zeros((100, 100), np.uint8)
    disc(mask, 30, 30, 10, 1)
    rules = {"fill_holes": [], "rules": [
        {"name": "a2", "class": 1, "action": "reclass", "to": 2,
         "conditions": [{"metric": "area", "op": "gt", "value": 50}]},
        {"name": "borra 2", "class": 2, "action": "delete",
         "conditions": [{"metric": "area", "op": "gt", "value": 50}]}]}
    reasons = []
    out, dropped = postprocess.apply(mask, rules, reasons=reasons)
    assert not out.any() and dropped[30, 30] == 1
    assert (reasons[0]["class"], reasons[0]["as_class"]) == (1, 2)
    rules["rules"].insert(1, {"name": "protege", "class": 2, "action": "keep", "conditions": []})
    out, dropped = postprocess.apply(mask, rules)
    assert out[30, 30] == 2 and not dropped.any()


def test_fill_holes_and_validation():
    mask = np.zeros((60, 60), np.uint8)
    mask[10:50, 10:50] = 2
    mask[25:35, 25:35] = 0
    out, _ = postprocess.apply(mask, {"rules": [], "fill_holes": [2]})
    assert out[30, 30] == 2
    for bad in ({}, {"rules": [{"conditions": [], "action": "borrar"}]},
                {"rules": [{"conditions": [{"metric": "area", "op": "gt", "value": "x"}], "action": "keep"}]}):
        with pytest.raises(ValueError):
            postprocess.validate(bad)


def test_store_is_global_and_migrates_the_old_per_backend_format(tmp_path):
    path = tmp_path / "p.json"
    assert PostprocessStore(str(path)).get() == postprocess.defaults()
    rules = {"rules": [{"name": "mía", "conditions": [], "action": "keep"}], "fill_holes": [1]}
    path.write_text('{"u-resnet": ' + __import__("json").dumps(rules) + "}")
    assert PostprocessStore(str(path)).get() == rules
    store = PostprocessStore(str(path))
    store.set(None)
    assert PostprocessStore(str(path)).get() == postprocess.defaults()
