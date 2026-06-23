"""Shared assertion helpers for the test suite."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

ATOL = 5e-7
RTOL = 0.0


def to_numpy(x: object) -> np.ndarray:
    """Convert any backend tensor to a NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, tf.Tensor):
        return x.numpy()
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def assert_allclose(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), atol=atol, rtol=rtol)


def assert_array_equal(actual, expected):
    np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))
