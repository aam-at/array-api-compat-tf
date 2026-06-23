"""Cross-backend tests comparing TensorFlow (via array_api_compat_tf) against
NumPy (via array_api_compat)."""

from __future__ import annotations

import array_api_compat
import numpy as np
import pytest
import tensorflow as tf

from array_api_compat_tf import array_namespace as tf_array_namespace
from tests.helpers import assert_allclose, to_numpy

pytestmark = pytest.mark.cross_backend

# ---------------------------------------------------------------------------
# Seed data (batched)
# ---------------------------------------------------------------------------

_BATCH_2D = np.array([[3.0, -1.0, 0.0, 2.5], [-4.0, 1.0, 5.0, -2.0]], dtype=np.float32)
_BATCH_3D = np.array([[[1.0, -2.0], [3.0, 4.0]], [[5.0, -6.0], [7.0, 8.0]]], dtype=np.float32)


# ===========================================================================
# asarray
# ===========================================================================


class TestAsarray:
    def test_from_nested_list(self, atol, rtol):
        data = [[1, 2], [3, 4]]
        xp_tf = tf_array_namespace(tf.constant(data, dtype=tf.float32))
        xp_np = array_api_compat.array_namespace(np.array(data, dtype=np.float32))

        out_tf = xp_tf.asarray(data)
        out_np = xp_np.asarray(data)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)


# ===========================================================================
# astype
# ===========================================================================


class TestAstype:
    def test_cast_float32_to_float64(self, atol, rtol):
        data = _BATCH_2D
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.astype(tf.constant(data), xp_tf.float64)
        out_np = xp_np.astype(data, xp_np.float64)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)
        assert to_numpy(out_tf).dtype == np.float64
        assert to_numpy(out_np).dtype == np.float64


# ===========================================================================
# clip
# ===========================================================================


class TestClip:
    def test_min_only(self, atol, rtol):
        data = _BATCH_2D
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.clip(tf.constant(data), min=0.0)
        out_np = xp_np.clip(data, min=0.0)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)

    def test_min_and_max(self, atol, rtol):
        data = _BATCH_2D
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.clip(tf.constant(data), min=-1.0, max=2.0)
        out_np = xp_np.clip(data, min=-1.0, max=2.0)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)


# ===========================================================================
# concat
# ===========================================================================


class TestConcat:
    def test_axis0(self, atol, rtol):
        a, b = _BATCH_2D[:1], _BATCH_2D[1:]
        xp_tf = tf_array_namespace(tf.constant(a))
        xp_np = array_api_compat.array_namespace(a)

        out_tf = xp_tf.concat([tf.constant(a), tf.constant(b)], axis=0)
        out_np = xp_np.concat([a, b], axis=0)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)

    def test_axis1(self, atol, rtol):
        a = _BATCH_2D[:, :2]
        b = _BATCH_2D[:, 2:]
        xp_tf = tf_array_namespace(tf.constant(a))
        xp_np = array_api_compat.array_namespace(a)

        out_tf = xp_tf.concat([tf.constant(a), tf.constant(b)], axis=1)
        out_np = xp_np.concat([a, b], axis=1)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)


# ===========================================================================
# cumulative_sum
# ===========================================================================


class TestCumulativeSum:
    def test_axis1(self, atol, rtol):
        data = _BATCH_2D
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.cumulative_sum(tf.constant(data), axis=1)
        out_np = xp_np.cumulative_sum(data, axis=1)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)

    def test_include_initial(self, atol, rtol):
        data = _BATCH_2D
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.cumulative_sum(tf.constant(data), axis=1, include_initial=True)
        out_np = xp_np.cumulative_sum(data, axis=1, include_initial=True)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)
        assert to_numpy(out_tf).shape == to_numpy(out_np).shape


# ===========================================================================
# searchsorted
# ===========================================================================


class TestSearchsorted:
    def test_searchsorted_batch_tf_works(self, xp, atol, rtol):
        """TF searchsorted supports batch dims; NumPy does not."""
        sorted_2d = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        values_2d = np.array([[2.0, 4.0], [3.0, 5.0]], dtype=np.float32)
        out_tf = xp.searchsorted(tf.constant(sorted_2d), tf.constant(values_2d), side="left")
        assert out_tf.shape == (2, 2)

    def test_searchsorted_numpy_no_batch(self):
        """Verify NumPy searchsorted rejects batch dimensions."""
        xp_np = array_api_compat.array_namespace(np.array([1.0]))
        sorted_2d = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        values_2d = np.array([[2.0, 4.0], [3.0, 5.0]], dtype=np.float32)
        with pytest.raises((ValueError, TypeError)):
            xp_np.searchsorted(sorted_2d, values_2d, side="left")


# ===========================================================================
# take
# ===========================================================================


class TestTake:
    def test_axis1(self, atol, rtol):
        data = _BATCH_2D
        idx = np.array([0, 2], dtype=np.int64)
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.take(tf.constant(data), tf.constant(idx), axis=1)
        out_np = xp_np.take(data, idx, axis=1)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)


# ===========================================================================
# where
# ===========================================================================


class TestWhere:
    def test_batch(self, atol, rtol):
        data = _BATCH_2D
        cond = (data > 0).astype(np.bool_)
        pos = np.ones_like(data)
        neg = np.full_like(data, -1.0)

        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.where(tf.constant(cond), tf.constant(pos), tf.constant(neg))
        out_np = xp_np.where(cond, pos, neg)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)


# ===========================================================================
# result_type
# ===========================================================================


class TestResultType:
    def test_promotion_f32_f64(self):
        f32 = np.array([[1.0]], dtype=np.float32)
        f64 = np.array([[2.0]], dtype=np.float64)

        xp_tf = tf_array_namespace(tf.constant(f32))
        xp_np = array_api_compat.array_namespace(f32)

        rt_tf = xp_tf.result_type(tf.constant(f32), tf.constant(f64))
        rt_np = xp_np.result_type(f32, f64)

        out_tf = to_numpy(xp_tf.asarray([0], dtype=rt_tf))
        out_np = to_numpy(xp_np.asarray([0], dtype=rt_np))
        assert out_tf.dtype == out_np.dtype == np.float64


# ===========================================================================
# linalg.vector_norm
# ===========================================================================


class TestVectorNorm:
    @pytest.mark.parametrize("ord_val", [1, 2, float("inf"), float("-inf")])
    def test_ord_values(self, ord_val, atol, rtol):
        data = _BATCH_2D
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.linalg.vector_norm(tf.constant(data), ord=ord_val, axis=1)
        out_np = xp_np.linalg.vector_norm(data, ord=ord_val, axis=1)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)

    def test_keepdims(self, atol, rtol):
        data = _BATCH_2D
        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        out_tf = xp_tf.linalg.vector_norm(tf.constant(data), ord=2, axis=1, keepdims=True)
        out_np = xp_np.linalg.vector_norm(data, ord=2, axis=1, keepdims=True)
        assert_allclose(out_tf, out_np, atol=atol, rtol=rtol)
        assert to_numpy(out_tf).shape == to_numpy(out_np).shape


# ===========================================================================
# End-to-end pipeline
# ===========================================================================


class TestEndToEndPipeline:
    def test_clip_norm_cumsum(self, atol, rtol):
        """Clip → vector_norm per-row → cumulative_sum on 2D batch."""
        data = _BATCH_2D

        xp_tf = tf_array_namespace(tf.constant(data))
        xp_np = array_api_compat.array_namespace(data)

        c_tf = xp_tf.clip(tf.constant(data), min=-1.0, max=2.0)
        c_np = xp_np.clip(data, min=-1.0, max=2.0)

        n_tf = xp_tf.linalg.vector_norm(c_tf, ord=2, axis=1)
        n_np = xp_np.linalg.vector_norm(c_np, ord=2, axis=1)

        cs_tf = xp_tf.cumulative_sum(n_tf, axis=0)
        cs_np = xp_np.cumulative_sum(n_np, axis=0)

        assert_allclose(cs_tf, cs_np, atol=atol, rtol=rtol)
