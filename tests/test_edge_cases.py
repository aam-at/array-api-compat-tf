"""Edge-case and error-path tests for patched namespace operations."""

import pytest
import tensorflow as tf

from tests.helpers import assert_allclose, assert_array_equal


class TestScalarTensors:
    def test_asarray_scalar(self, xp):
        out = xp.asarray(1.0)
        assert isinstance(out, tf.Tensor)
        assert out.shape == ()

    def test_astype_scalar(self, xp):
        t = tf.constant(1.0, dtype=tf.float32)
        out = xp.astype(t, tf.float64)
        assert out.dtype == tf.float64
        assert out.shape == ()

    def test_clip_scalar(self, xp, atol, rtol):
        t = tf.constant(5.0)
        out = xp.clip(t, min=0.0, max=3.0)
        assert_allclose(out, 3.0, atol=atol, rtol=rtol)

    def test_cumulative_sum_scalar_no_axis(self, xp):
        t = tf.constant(7.0)
        out = xp.cumulative_sum(t)
        assert_array_equal(out, [7.0])

    def test_cumulative_sum_scalar_include_initial(self, xp):
        t = tf.constant(7.0)
        out = xp.cumulative_sum(t, include_initial=True)
        assert_array_equal(out, [0.0, 7.0])

    def test_vector_norm_scalar(self, xp, atol, rtol):
        t = tf.constant(-3.0)
        out = xp.linalg.vector_norm(t, ord=2)
        assert_allclose(out, 3.0, atol=atol, rtol=rtol)


class TestEmptyTensors:
    def test_asarray_empty(self, xp):
        out = xp.asarray([])
        assert isinstance(out, tf.Tensor)
        assert out.shape == (0,)

    def test_concat_empty(self, xp):
        a = tf.constant([1.0, 2.0])
        b = tf.constant([], dtype=tf.float32)
        out = xp.concat([a, b], axis=0)
        assert_array_equal(out, [1.0, 2.0])

    def test_clip_empty(self, xp):
        t = tf.constant([], dtype=tf.float32)
        out = xp.clip(t, min=0.0)
        assert out.shape == (0,)

    def test_cumulative_sum_empty(self, xp):
        t = tf.constant([], dtype=tf.float32)
        out = xp.cumulative_sum(t, axis=0)
        assert out.shape == (0,)

    def test_cumulative_sum_empty_include_initial(self, xp):
        t = tf.constant([], dtype=tf.float32)
        out = xp.cumulative_sum(t, axis=0, include_initial=True)
        assert out.shape == (1,)
        assert_array_equal(out, [0.0])


class TestDtypePreservation:
    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64, tf.int32, tf.int64])
    def test_asarray_dtype(self, xp, dtype):
        out = xp.asarray([1, 2, 3], dtype=dtype)
        assert out.dtype == dtype

    @pytest.mark.parametrize(
        "src_dtype,dst_dtype",
        [
            (tf.float32, tf.float64),
            (tf.float64, tf.float32),
            (tf.int32, tf.float32),
            (tf.int64, tf.int32),
        ],
    )
    def test_astype_conversion(self, xp, src_dtype, dst_dtype):
        t = tf.constant([1, 2, 3], dtype=src_dtype)
        out = xp.astype(t, dst_dtype)
        assert out.dtype == dst_dtype

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_cumulative_sum_preserves_dtype(self, xp, dtype):
        t = tf.constant([1.0, 2.0, 3.0], dtype=dtype)
        out = xp.cumulative_sum(t, axis=0)
        assert out.dtype == dtype

    def test_cumulative_sum_dtype_override(self, xp):
        t = tf.constant([1.0, 2.0], dtype=tf.float32)
        out = xp.cumulative_sum(t, axis=0, dtype=tf.float64)
        assert out.dtype == tf.float64

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_clip_preserves_dtype(self, xp, dtype):
        t = tf.constant([1.0, 2.0, 3.0], dtype=dtype)
        out = xp.clip(t, min=0.0, max=5.0)
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_concat_preserves_dtype(self, xp, dtype):
        a = tf.constant([1.0], dtype=dtype)
        b = tf.constant([2.0], dtype=dtype)
        out = xp.concat([a, b], axis=0)
        assert out.dtype == dtype


class TestMultiDimensionalInputs:
    def test_cumulative_sum_2d_axis0(self, xp):
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        out = xp.cumulative_sum(t, axis=0)
        assert_array_equal(out, [[1.0, 2.0], [4.0, 6.0]])

    def test_cumulative_sum_2d_axis0_include_initial(self, xp):
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        out = xp.cumulative_sum(t, axis=0, include_initial=True)
        assert out.shape == (3, 2)
        assert_array_equal(out, [[0.0, 0.0], [1.0, 2.0], [4.0, 6.0]])

    def test_cumulative_sum_no_axis_flattens(self, xp):
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        out = xp.cumulative_sum(t)
        assert_array_equal(out, [1.0, 3.0, 6.0, 10.0])

    def test_vector_norm_batched_2d(self, xp, atol, rtol):
        t = tf.constant([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
        out = xp.linalg.vector_norm(t, ord=2, axis=1)
        assert_allclose(out, [1.0, 1.0, 5.0], atol=atol, rtol=rtol)

    def test_vector_norm_3d_batched(self, xp, atol, rtol):
        t = tf.constant([[[3.0, 4.0], [5.0, 12.0]]])
        out = xp.linalg.vector_norm(t, ord=2, axis=-1)
        assert_allclose(out, [[5.0, 13.0]], atol=atol, rtol=rtol)

    def test_searchsorted_batched(self, xp):
        x = tf.constant([1.0, 3.0, 5.0, 7.0, 9.0])
        v = tf.constant([0.0, 1.0, 4.0, 9.0, 10.0])
        out = xp.searchsorted(x, v, side="left")
        assert_array_equal(out, [0, 0, 2, 4, 5])


class TestErrorPaths:
    def test_searchsorted_invalid_side(self, xp):
        x = tf.constant([1.0, 2.0, 3.0])
        v = tf.constant([1.5])
        with pytest.raises(ValueError, match="'side' must be 'left' or 'right'"):
            xp.searchsorted(x, v, side="middle")

    def test_vector_norm_ord_zero(self, xp):
        t = tf.constant([1.0, 2.0])
        with pytest.raises(NotImplementedError, match="ord=0"):
            xp.linalg.vector_norm(t, ord=0, axis=0)

    def test_vector_norm_ord_unsupported_type(self, xp):
        t = tf.constant([1.0, 2.0])
        with pytest.raises(NotImplementedError, match="does not support ord="):
            xp.linalg.vector_norm(t, ord="fro", axis=0)

    def test_clip_no_args_raises_typeerror(self, xp):
        t = tf.constant([1.0])
        with pytest.raises(TypeError, match="requires at least one"):
            xp.clip(t)
