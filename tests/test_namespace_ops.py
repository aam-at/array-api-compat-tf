"""Tests for patched Array API operations on the TensorFlow namespace."""

import math

import pytest
import tensorflow as tf

from tests.helpers import assert_allclose, assert_array_equal


class TestAsarray:
    def test_from_list(self, xp):
        out = xp.asarray([1.0, 2.0, 3.0])
        assert isinstance(out, tf.Tensor)
        assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_with_dtype(self, xp):
        out = xp.asarray([1, 2, 3], dtype=tf.float32)
        assert out.dtype == tf.float32

    def test_copy_none(self, xp):
        t = tf.constant([1.0, 2.0])
        out = xp.asarray(t, copy=None)
        assert isinstance(out, tf.Tensor)

    def test_copy_true(self, xp):
        t = tf.constant([1.0, 2.0])
        out = xp.asarray(t, copy=True)
        assert isinstance(out, tf.Tensor)


class TestAstype:
    def test_cast(self, xp):
        t = tf.constant([1.0, 2.0], dtype=tf.float32)
        out = xp.astype(t, tf.float64)
        assert out.dtype == tf.float64
        assert_array_equal(out, [1.0, 2.0])

    def test_no_copy_same_dtype(self, xp):
        t = tf.constant([1.0], dtype=tf.float32)
        out = xp.astype(t, tf.float32, copy=False)
        assert out is t


class TestAstypeCopyBehavior:
    def test_copy_true_same_dtype_returns_new(self, xp):
        t = tf.constant([1.0, 2.0], dtype=tf.float32)
        out = xp.astype(t, tf.float32, copy=True)
        assert out.dtype == tf.float32
        assert_array_equal(out, [1.0, 2.0])

    def test_copy_false_same_dtype_returns_same(self, xp):
        t = tf.constant([1.0, 2.0], dtype=tf.float32)
        out = xp.astype(t, tf.float32, copy=False)
        assert out is t

    def test_copy_false_different_dtype_casts(self, xp):
        t = tf.constant([1.0, 2.0], dtype=tf.float32)
        out = xp.astype(t, tf.float64, copy=False)
        assert out.dtype == tf.float64
        assert out is not t

    def test_copy_default_is_true(self, xp):
        t = tf.constant([1.0], dtype=tf.float32)
        out = xp.astype(t, tf.float32)
        assert out.dtype == tf.float32
        assert_array_equal(out, [1.0])


class TestClip:
    def test_min_only(self, xp):
        t = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = xp.clip(t, min=0.0)
        assert_array_equal(out, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_max_only(self, xp):
        t = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = xp.clip(t, max=1.0)
        assert_array_equal(out, [-2.0, -1.0, 0.0, 1.0, 1.0])

    def test_min_and_max(self, xp):
        t = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = xp.clip(t, min=-1.0, max=1.0)
        assert_array_equal(out, [-1.0, -1.0, 0.0, 1.0, 1.0])

    def test_no_args_raises(self, xp):
        t = tf.constant([1.0])
        with pytest.raises(TypeError):
            xp.clip(t)


class TestConcat:
    def test_basic(self, xp):
        a = tf.constant([1.0, 2.0])
        b = tf.constant([3.0, 4.0])
        out = xp.concat([a, b], axis=0)
        assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])

    def test_2d(self, xp):
        a = tf.constant([[1.0], [2.0]])
        b = tf.constant([[3.0], [4.0]])
        out = xp.concat([a, b], axis=1)
        assert_array_equal(out, [[1.0, 3.0], [2.0, 4.0]])


class TestCumulativeSum:
    def test_basic(self, xp):
        t = tf.constant([1.0, 2.0, 3.0])
        out = xp.cumulative_sum(t, axis=0)
        assert_array_equal(out, [1.0, 3.0, 6.0])

    def test_include_initial(self, xp):
        t = tf.constant([1.0, 2.0, 3.0])
        out = xp.cumulative_sum(t, axis=0, include_initial=True)
        assert_array_equal(out, [0.0, 1.0, 3.0, 6.0])

    def test_2d(self, xp):
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        out = xp.cumulative_sum(t, axis=1, include_initial=True)
        assert out.shape == (2, 3)
        assert_array_equal(out, [[0.0, 1.0, 3.0], [0.0, 3.0, 7.0]])


class TestSearchsorted:
    def test_right(self, xp):
        x = tf.constant([1.0, 3.0, 5.0, 7.0])
        v = tf.constant([2.0, 4.0, 6.0])
        out = xp.searchsorted(x, v, right=True)
        assert_array_equal(out, [1, 2, 3])

    def test_left(self, xp):
        x = tf.constant([1.0, 3.0, 5.0, 7.0])
        v = tf.constant([3.0, 5.0])
        out = xp.searchsorted(x, v, right=False)
        assert_array_equal(out, [1, 2])

    def test_side_kwarg(self, xp):
        x = tf.constant([1.0, 3.0, 5.0])
        v = tf.constant([3.0])
        out_left = xp.searchsorted(x, v, side="left")
        out_right = xp.searchsorted(x, v, side="right")
        assert out_left.numpy()[0] == 1
        assert out_right.numpy()[0] == 2


class TestVectorNorm:
    def test_l1(self, xp, atol, rtol):
        t = tf.constant([1.0, -2.0, 3.0])
        out = xp.linalg.vector_norm(t, ord=1, axis=0)
        assert_allclose(out, 6.0, atol=atol, rtol=rtol)

    def test_l2(self, xp, atol, rtol):
        t = tf.constant([3.0, 4.0])
        out = xp.linalg.vector_norm(t, ord=2, axis=0)
        assert_allclose(out, 5.0, atol=atol, rtol=rtol)

    def test_inf(self, xp, atol, rtol):
        t = tf.constant([1.0, -5.0, 3.0])
        out = xp.linalg.vector_norm(t, ord=math.inf, axis=0)
        assert_allclose(out, 5.0, atol=atol, rtol=rtol)

    def test_neg_inf(self, xp, atol, rtol):
        t = tf.constant([1.0, -5.0, 3.0])
        out = xp.linalg.vector_norm(t, ord=-math.inf, axis=0)
        assert_allclose(out, 1.0, atol=atol, rtol=rtol)

    def test_p_norm(self, xp, atol, rtol):
        t = tf.constant([1.0, 2.0, 3.0])
        out = xp.linalg.vector_norm(t, ord=3, axis=0)
        expected = (1.0**3 + 2.0**3 + 3.0**3) ** (1.0 / 3)
        assert_allclose(out, expected, atol=atol, rtol=rtol)

    def test_2d_axis(self, xp, atol, rtol):
        t = tf.constant([[3.0, 4.0], [5.0, 12.0]])
        out = xp.linalg.vector_norm(t, ord=2, axis=1)
        assert_allclose(out, [5.0, 13.0], atol=atol, rtol=rtol)

    def test_keepdims(self, xp):
        t = tf.constant([[3.0, 4.0]])
        out = xp.linalg.vector_norm(t, ord=2, axis=1, keepdims=True)
        assert out.shape == (1, 1)


class TestTake:
    def test_basic(self, xp):
        t = tf.constant([10.0, 20.0, 30.0, 40.0])
        idx = tf.constant([0, 2, 3])
        out = xp.take(t, idx, axis=0)
        assert_array_equal(out, [10.0, 30.0, 40.0])

    def test_no_axis_flattens(self, xp):
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        idx = tf.constant([0, 3])
        out = xp.take(t, idx)
        assert_array_equal(out, [1.0, 4.0])

    def test_2d_axis1(self, xp):
        t = tf.constant([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        idx = tf.constant([0, 2])
        out = xp.take(t, idx, axis=1)
        assert_array_equal(out, [[10.0, 30.0], [40.0, 60.0]])


class TestWhere:
    def test_basic(self, xp):
        cond = tf.constant([True, False, True])
        x1 = tf.constant([1.0, 2.0, 3.0])
        x2 = tf.constant([10.0, 20.0, 30.0])
        out = xp.where(cond, x1, x2)
        assert_array_equal(out, [1.0, 20.0, 3.0])


class TestResultType:
    def test_same_dtype(self, xp):
        a = tf.constant([1.0], dtype=tf.float32)
        result = xp.result_type(a)
        assert result == tf.float32

    def test_promotion(self, xp):
        a = tf.constant([1.0], dtype=tf.float32)
        b = tf.constant([1.0], dtype=tf.float64)
        result = xp.result_type(a, b)
        assert result == tf.float64


class TestDtypeAttributes:
    def test_float32(self, xp):
        assert xp.float32 == tf.float32

    def test_float64(self, xp):
        assert xp.float64 == tf.float64

    def test_int32(self, xp):
        assert xp.int32 == tf.int32

    def test_int64(self, xp):
        assert xp.int64 == tf.int64

    def test_bool(self, xp):
        assert xp.bool == tf.bool
