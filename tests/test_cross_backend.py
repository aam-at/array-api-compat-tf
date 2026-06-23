"""Cross-backend tests comparing TensorFlow vs PyTorch and TensorFlow vs JAX.

Every test creates the same logical input in each backend, runs the same
Array API operation through ``array_namespace()``, and asserts the results
match (converted to NumPy for comparison).
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from array_api_compat_tf import array_namespace, device, size
from tests.helpers import assert_allclose, assert_array_equal, to_numpy

pytestmark = pytest.mark.cross_backend

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED_DATA = np.array([3.0, -1.0, 0.0, 2.5, -4.0], dtype=np.float32)
_SEED_2D = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)


# ===========================================================================
# asarray / astype
# ===========================================================================


class TestAsarray:
    def test_from_list(self, peer_backend, atol, rtol):
        name, make = peer_backend
        data = [1.0, 2.0, 3.0]

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(np.array(data, dtype=np.float32)))

        out_tf = xp_tf.asarray(data)
        out_peer = xp_peer.asarray(data)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)


class TestAstype:
    def test_cast_float64(self, peer_backend, atol, rtol):
        name, make = peer_backend
        if name == "jax":
            pytest.skip("JAX x64 mode not enabled")
        data = _SEED_DATA

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.astype(tf.constant(data), xp_tf.float64)
        out_peer = xp_peer.astype(make(data), xp_peer.float64)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)
        assert to_numpy(out_tf).dtype == np.float64
        assert to_numpy(out_peer).dtype == np.float64


# ===========================================================================
# clip
# ===========================================================================


class TestClip:
    def test_min_only(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = _SEED_DATA

        out_tf = array_namespace(tf.constant(data)).clip(tf.constant(data), min=0.0)
        out_peer = array_namespace(make(data)).clip(make(data), min=0.0)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)

    def test_max_only(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = _SEED_DATA

        out_tf = array_namespace(tf.constant(data)).clip(tf.constant(data), max=1.0)
        out_peer = array_namespace(make(data)).clip(make(data), max=1.0)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)

    def test_min_and_max(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = _SEED_DATA

        out_tf = array_namespace(tf.constant(data)).clip(tf.constant(data), min=-1.0, max=2.0)
        out_peer = array_namespace(make(data)).clip(make(data), min=-1.0, max=2.0)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)


# ===========================================================================
# concat
# ===========================================================================


class TestConcat:
    def test_1d(self, peer_backend, atol, rtol):
        _, make = peer_backend
        a, b = _SEED_DATA[:3], _SEED_DATA[3:]

        xp_tf = array_namespace(tf.constant(a))
        xp_peer = array_namespace(make(a))

        out_tf = xp_tf.concat([tf.constant(a), tf.constant(b)], axis=0)
        out_peer = xp_peer.concat([make(a), make(b)], axis=0)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)

    def test_2d_axis1(self, peer_backend, atol, rtol):
        _, make = peer_backend
        a = _SEED_2D[:, :2]
        b = _SEED_2D[:, 2:]

        xp_tf = array_namespace(tf.constant(a))
        xp_peer = array_namespace(make(a))

        out_tf = xp_tf.concat([tf.constant(a), tf.constant(b)], axis=1)
        out_peer = xp_peer.concat([make(a), make(b)], axis=1)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)


# ===========================================================================
# cumulative_sum
# ===========================================================================


class TestCumulativeSum:
    def test_basic(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.cumulative_sum(tf.constant(data), axis=0)
        out_peer = xp_peer.cumulative_sum(make(data), axis=0)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)

    def test_include_initial(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.cumulative_sum(tf.constant(data), axis=0, include_initial=True)
        out_peer = xp_peer.cumulative_sum(make(data), axis=0, include_initial=True)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)
        assert to_numpy(out_tf).shape == to_numpy(out_peer).shape  # (4,) — one extra element


# ===========================================================================
# searchsorted
# ===========================================================================


class TestSearchsorted:
    def test_basic(self, peer_backend):
        _, make = peer_backend
        sorted_arr = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
        values = np.array([2.0, 4.0, 6.0], dtype=np.float32)

        xp_tf = array_namespace(tf.constant(sorted_arr))
        xp_peer = array_namespace(make(sorted_arr))

        out_tf = xp_tf.searchsorted(tf.constant(sorted_arr), tf.constant(values), side="left")
        out_peer = xp_peer.searchsorted(make(sorted_arr), make(values), side="left")
        assert_array_equal(out_tf, out_peer)

    def test_right(self, peer_backend):
        _, make = peer_backend
        sorted_arr = np.array([1.0, 3.0, 3.0, 5.0], dtype=np.float32)
        values = np.array([3.0], dtype=np.float32)

        xp_tf = array_namespace(tf.constant(sorted_arr))
        xp_peer = array_namespace(make(sorted_arr))

        out_tf = xp_tf.searchsorted(tf.constant(sorted_arr), tf.constant(values), side="right")
        out_peer = xp_peer.searchsorted(make(sorted_arr), make(values), side="right")
        assert_array_equal(out_tf, out_peer)


# ===========================================================================
# take
# ===========================================================================


class TestTake:
    def test_axis0(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = _SEED_DATA
        idx_np = np.array([0, 2, 4], dtype=np.int64)

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.take(tf.constant(data), tf.constant(idx_np), axis=0)
        out_peer = xp_peer.take(make(data), make(idx_np), axis=0)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)

    def test_2d_axis1(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = _SEED_2D
        idx_np = np.array([0, 2], dtype=np.int64)

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.take(tf.constant(data), tf.constant(idx_np), axis=1)
        out_peer = xp_peer.take(make(data), make(idx_np), axis=1)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)


# ===========================================================================
# where
# ===========================================================================


class TestWhere:
    def test_basic(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = _SEED_DATA
        cond_np = (data > 0).astype(np.bool_)
        pos = np.ones_like(data)
        neg = np.full_like(data, -1.0)

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.where(tf.constant(cond_np), tf.constant(pos), tf.constant(neg))
        out_peer = xp_peer.where(make(cond_np), make(pos), make(neg))
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)


# ===========================================================================
# result_type
# ===========================================================================


class TestResultType:
    def test_same_dtype_preserved(self, peer_backend):
        """result_type of same-dtype arrays returns that dtype."""
        _, make = peer_backend
        a = np.array([1.0], dtype=np.float32)

        xp_tf = array_namespace(tf.constant(a))
        xp_peer = array_namespace(make(a))

        rt_tf = xp_tf.result_type(tf.constant(a))
        rt_peer = xp_peer.result_type(make(a))

        out_tf = to_numpy(xp_tf.asarray([0], dtype=rt_tf))
        out_peer = to_numpy(xp_peer.asarray([0], dtype=rt_peer))
        assert out_tf.dtype == np.float32
        assert out_peer.dtype == np.float32


# ===========================================================================
# linalg.vector_norm
# ===========================================================================


class TestVectorNorm:
    @pytest.mark.parametrize("ord_val", [1, 2, float("inf"), float("-inf")])
    def test_ord_values(self, peer_backend, ord_val, atol, rtol):
        _, make = peer_backend
        data = _SEED_DATA

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.linalg.vector_norm(tf.constant(data), ord=ord_val, axis=0)
        out_peer = xp_peer.linalg.vector_norm(make(data), ord=ord_val, axis=0)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)

    def test_keepdims(self, peer_backend, atol, rtol):
        _, make = peer_backend
        data = _SEED_2D

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        out_tf = xp_tf.linalg.vector_norm(tf.constant(data), ord=2, axis=1, keepdims=True)
        out_peer = xp_peer.linalg.vector_norm(make(data), ord=2, axis=1, keepdims=True)
        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)
        assert to_numpy(out_tf).shape == to_numpy(out_peer).shape


# ===========================================================================
# dtype attributes
# ===========================================================================


class TestDtypeAttributes:
    @pytest.mark.parametrize("dtype_name", ["float32", "float64", "int32", "int64"])
    def test_dtype_matches(self, peer_backend, dtype_name):
        """Namespace dtype attributes resolve to the same NumPy dtype."""
        name, make = peer_backend
        # JAX without x64 mode truncates 64-bit dtypes to 32-bit
        if name == "jax" and "64" in dtype_name:
            pytest.skip("JAX x64 mode not enabled")

        dummy = np.array([1.0], dtype=np.float32)

        xp_tf = array_namespace(tf.constant(dummy))
        xp_peer = array_namespace(make(dummy))

        dt_tf = getattr(xp_tf, dtype_name)
        dt_peer = getattr(xp_peer, dtype_name)

        # Create a tensor with that dtype and compare the numpy dtype
        out_tf = to_numpy(xp_tf.asarray([1], dtype=dt_tf))
        out_peer = to_numpy(xp_peer.asarray([1], dtype=dt_peer))
        assert out_tf.dtype == out_peer.dtype


# ===========================================================================
# Compat helpers: device, size
# ===========================================================================


class TestSizeCrossBackend:
    def test_1d(self, peer_backend):
        _, make = peer_backend
        data = _SEED_DATA
        assert size(tf.constant(data)) == size(make(data))

    def test_2d(self, peer_backend):
        _, make = peer_backend
        data = _SEED_2D
        assert size(tf.constant(data)) == size(make(data))

    def test_scalar(self, peer_backend):
        _, make = peer_backend
        assert size(tf.constant(1.0)) == size(make(np.float32(1.0)))


class TestDeviceCrossBackend:
    def test_returns_something(self, peer_backend):
        """Both backends return a non-None device."""
        _, make = peer_backend
        data = np.array([1.0], dtype=np.float32)
        assert device(tf.constant(data)) is not None
        assert device(make(data)) is not None


# ===========================================================================
# End-to-end: multi-step pipeline
# ===========================================================================


class TestEndToEndPipeline:
    """Run a small multi-step Array API pipeline across backends and
    compare."""

    def test_clip_norm_cumsum(self, peer_backend, atol, rtol):
        """Clip → vector_norm per-row → cumulative_sum pipeline."""
        _, make = peer_backend
        data = _SEED_2D  # (2, 3)

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        # Step 1: clip to [−1, 2]
        c_tf = xp_tf.clip(tf.constant(data), min=-1.0, max=2.0)
        c_peer = xp_peer.clip(make(data), min=-1.0, max=2.0)

        # Step 2: L2 norm per row
        n_tf = xp_tf.linalg.vector_norm(c_tf, ord=2, axis=1)
        n_peer = xp_peer.linalg.vector_norm(c_peer, ord=2, axis=1)

        # Step 3: cumulative sum
        cs_tf = xp_tf.cumulative_sum(n_tf, axis=0)
        cs_peer = xp_peer.cumulative_sum(n_peer, axis=0)

        assert_allclose(cs_tf, cs_peer, atol=atol, rtol=rtol)

    def test_take_where_concat(self, peer_backend, atol, rtol):
        """Take → where → concat pipeline."""
        _, make = peer_backend
        data = _SEED_DATA
        idx_np = np.array([0, 2, 4], dtype=np.int64)

        xp_tf = array_namespace(tf.constant(data))
        xp_peer = array_namespace(make(data))

        # Step 1: take indices
        t_tf = xp_tf.take(tf.constant(data), tf.constant(idx_np), axis=0)
        t_peer = xp_peer.take(make(data), make(idx_np), axis=0)

        # Step 2: where — replace negatives with 0
        zero_tf = xp_tf.asarray(np.zeros(3, dtype=np.float32))
        zero_peer = xp_peer.asarray(np.zeros(3, dtype=np.float32))
        cond_tf = t_tf > zero_tf
        cond_peer = t_peer > zero_peer
        w_tf = xp_tf.where(cond_tf, t_tf, zero_tf)
        w_peer = xp_peer.where(cond_peer, t_peer, zero_peer)

        # Step 3: concat with original
        out_tf = xp_tf.concat([xp_tf.asarray(data), w_tf], axis=0)
        out_peer = xp_peer.concat([xp_peer.asarray(data), w_peer], axis=0)

        assert_allclose(out_tf, out_peer, atol=atol, rtol=rtol)
