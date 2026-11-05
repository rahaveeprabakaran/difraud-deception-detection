import numpy as np

from src.shortcut import fit_score, matched_random_words, strip_words, top_unigrams


def _toy():
    rng = np.random.default_rng(0)
    base = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta", "zeta", "iota"]
    x, y = [], []
    for i in range(300):
        label = i % 2
        words = list(rng.choice(base, 6)) + (["shortcut"] if label else [])
        x.append(" ".join(words))
        y.append(label)
    return x, y


def test_strip_words_removes_whole_words_only():
    out = strip_words(["The Obamacare plan", "obama care", "obamacares"], ["obamacare"])
    assert out[0].split() == ["The", "plan"] and out[1] == "obama care" and out[2] == "obamacares"


def test_top_word_is_the_planted_shortcut_and_removal_hurts():
    x, y = _toy()
    xtr, ytr, xte, yte = x[:200], y[:200], x[200:], y[200:]
    f1, auc, vec, clf, A = fit_score(xtr, ytr, xte, yte)
    words = top_unigrams(vec, clf, 1)
    assert words == ["shortcut"] and auc > 0.99
    _, auc2, *_ = fit_score(strip_words(xtr, words), ytr, strip_words(xte, words), yte)
    assert auc2 < 0.7


def test_matched_random_words_exclude_removed_and_have_similar_frequency():
    x, y = _toy()
    _, _, vec, clf, A = fit_score(x, y, x, y)
    rng = np.random.default_rng(1)
    ctrl = matched_random_words(vec, A, ["alpha"], 3, rng)
    assert len(set(ctrl)) == 3
    assert "alpha" not in ctrl and len(ctrl) == 3 and all(" " not in w for w in ctrl)
