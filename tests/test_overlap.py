from src.overlap import analyze


def test_counts_and_label_conflicts():
    trx = ["a b", "a b", "c", "Hello  World http://x.io/1", "d"]
    try_ = [1, 0, 1, 1, 0]
    tex = ["a b", "c", "hello world http://y.io/2", "zzz"]
    tey = [1, 0, 1, 0]
    r = analyze(trx, try_, tex, tey)
    e, n = r["exact"], r["normalized"]
    assert e["test_in_train"] == 2 and e["test_in_train_label_conflict"] == 2
    assert e["train_extra_copies"] == 1 and e["train_groups_label_conflict"] == 1
    assert n["test_in_train"] == 3 and n["test_in_train_label_agrees"] == 1
