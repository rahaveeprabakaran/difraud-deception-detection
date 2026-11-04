import json, tempfile, os
from src.data import read_jsonl, check_splits

def w(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows: f.write(json.dumps(r) + "\n")
    f.close(); return f.name

def raises(rows):
    p = w(rows)
    try:
        read_jsonl(p)
    except ValueError:
        return True
    finally:
        os.remove(p)
    return False

def test_bad_inputs_rejected():
    assert raises([{"text": "hi", "label": 2}])
    assert raises([{"text": "hi", "label": True}])
    assert raises([{"text": "   ", "label": 1}])
    assert raises([{"text": None, "label": 1}])
    assert raises([{"label": 1}])

def test_good_input_and_leak_report():
    p = w([{"text": "a", "label": 0}, {"text": "b", "label": 1}])
    t, y = read_jsonl(p); os.remove(p)
    assert t == ["a", "b"] and y == [0, 1]
    rep = check_splits({"train": (["a", "b"], [0, 1]), "validation": (["c", "d"], [0, 1]),
                        "test": (["a", "e"], [0, 1])})
    assert rep["leak_train_test"] == 1 and rep["leak_train_val"] == 0

def test_long_text_allowed():
    p = w([{"text": "a" * 2_000_000, "label": 1}])
    t, y = read_jsonl(p); os.remove(p)
    assert len(t[0]) == 2_000_000
