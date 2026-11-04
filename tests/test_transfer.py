import json
import numpy as np
import src.transfer as T

TEST = {"a": (["x1", "x2", "shared", "x4"], [1, 0, 1, 0]),
        "b": (["y1", "y2", "y3", "y4"], [1, 0, 1, 0])}


def fake_env(monkeypatch):
    monkeypatch.setattr(T, "get_path", lambda d, s, m: (d, s))
    monkeypatch.setattr(T, "read_jsonl", lambda p: TEST[p[0]])


def test_scores_only_other_domains_and_no_texts(monkeypatch):
    fake_env(monkeypatch)
    seen = []
    def predict(texts, labels):
        seen.append(list(texts))
        return [0.9 if l else 0.1 for l in labels]
    out = T.score_targets("b", ["train1", "shared"], {}, predict, domains=["a", "b"])
    assert list(out) == ["a"]                       # source domain is skipped
    assert out["a"]["overlap_idx"] == [2]           # the shared text is found
    assert "x1" not in json.dumps(out)              # no texts are stored
    assert seen == [["x1", "x2", "shared", "x4"]]   # only the target test texts were scored


def test_build_excludes_overlap(tmp_path):
    y = [1, 0, 1, 0]
    res = {"meta": {"domain": "b"}, "validation": {"f1": 0.5},
           "predictions": {"test": {"y": y, "p1": [0.9, 0.1, 0.8, 0.2]}},
           "transfer": {"a": {"y": y, "p1": [0.9, 0.1, 0.1, 0.2], "overlap_idx": [2]}}}
    (tmp_path / "r.json").write_text(json.dumps(res))
    cells, overlap, _ = T.build(tmp_path)
    assert cells[("b", "a")]["auc"] < 1.0
    assert overlap[("b", "a")][0] == 1 and overlap[("b", "a")][1]["auc"] == 1.0
