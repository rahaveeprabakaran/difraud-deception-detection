import numpy as np

from src.blend import blend, choose_weight


def test_blend_is_a_weighted_average():
    assert np.allclose(blend([1.0, 0.0], [0.0, 1.0], 0.25), [0.25, 0.75])


def test_weight_follows_the_better_model():
    y = np.array([1, 1, 0, 0])
    good = np.array([0.9, 0.8, 0.1, 0.2])
    bad = np.array([0.1, 0.2, 0.9, 0.8])
    assert choose_weight(y, good, bad)[0] == 1.0
    assert choose_weight(y, bad, good)[0] <= 0.4  # the good model has most of the weight


def test_tie_goes_to_the_transformer():
    y = np.array([1, 0])
    same = np.array([0.9, 0.1])
    assert choose_weight(y, same, same)[0] == 1.0
