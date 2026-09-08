"""``sim/isaac/world._resample_nearest`` — how a smaller depth frame is served
from the RGB camera's render (numpy only, no Isaac)."""

from __future__ import annotations

import numpy as np

from embodiedscore_envs.benchmarks.env.sim.isaac.world import _resample_nearest


def test_identity_when_sizes_match():
    f = np.arange(16.0).reshape(4, 4)
    assert _resample_nearest(f, 4, 4) is f


def test_pixel_centres_integer_ratio():
    f = np.arange(64.0).reshape(8, 8)
    out = _resample_nearest(f, 4, 4)
    # output pixel (i, j) covers rows 2i..2i+1; its centre (2i + 1) is the sampled source pixel
    assert out.shape == (4, 4)
    assert np.array_equal(out, f[1::2, 1::2])
    assert set(out.ravel()) <= set(f.ravel())      # never an average


def test_non_integer_ratio_stays_in_bounds_and_monotone():
    f = np.arange(100.0).reshape(10, 10)
    out = _resample_nearest(f, 3, 7)
    assert out.shape == (3, 7)
    assert (np.diff(out, axis=0) >= 0).all() and (np.diff(out, axis=1) >= 0).all()
    assert out.max() <= 99.0 and out.min() >= 0.0
