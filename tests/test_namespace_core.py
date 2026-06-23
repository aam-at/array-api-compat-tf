"""Tests for namespace discovery, fallthrough, and required-op probing."""

import pytest
import tensorflow as tf

from array_api_compat_tf import (
    TensorFlowArrayApiNamespace,
    get_namespace,
    namespace_supports_required_ops,
)
from tests.helpers import assert_allclose, assert_array_equal


class TestGetNamespace:
    def test_returns_namespace(self):
        ns = get_namespace()
        assert isinstance(ns, TensorFlowArrayApiNamespace)

    def test_cached(self):
        assert get_namespace() is get_namespace()


class TestFallthrough:
    def test_abs(self, xp):
        t = tf.constant([-1.0, 2.0, -3.0])
        out = xp.abs(t)
        assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_sum(self, xp, atol, rtol):
        t = tf.constant([1.0, 2.0, 3.0])
        out = xp.sum(t)
        assert_allclose(out, 6.0, atol=atol, rtol=rtol)

    def test_zeros_like(self, xp):
        t = tf.constant([1.0, 2.0])
        out = xp.zeros_like(t)
        assert_array_equal(out, [0.0, 0.0])

    def test_missing_raises(self, xp):
        with pytest.raises(NotImplementedError):
            xp.nonexistent_function_xyz()


class TestGetAttrFallback:
    def test_existing_tfnp_op(self, xp, atol, rtol):
        t = tf.constant([0.0])
        out = xp.sin(t)
        assert_allclose(out, [0.0], atol=atol, rtol=rtol)

    def test_existing_tfnp_op_exp(self, xp, atol, rtol):
        t = tf.constant([0.0])
        out = xp.exp(t)
        assert_allclose(out, [1.0], atol=atol, rtol=rtol)

    def test_missing_op_raises_not_implemented(self, xp):
        with pytest.raises(NotImplementedError, match="is not available"):
            xp.this_op_definitely_does_not_exist_xyz()

    def test_missing_op_message_contains_name(self, xp):
        with pytest.raises(NotImplementedError, match="'totally_bogus_operation_42'"):
            xp.totally_bogus_operation_42()


class TestLinalgGetAttrFallback:
    def test_missing_linalg_op_raises(self, xp):
        with pytest.raises(NotImplementedError, match="is not available"):
            xp.linalg.nonexistent_linalg_op_xyz()

    def test_missing_linalg_op_message_contains_name(self, xp):
        with pytest.raises(NotImplementedError, match="'bogus_linalg_function_42'"):
            xp.linalg.bogus_linalg_function_42()


class TestReprMethods:
    def test_namespace_repr(self, xp):
        r = repr(xp)
        assert "TensorFlowArrayApiNamespace" in r
        assert "tf" in r

    def test_linalg_repr(self, xp):
        r = repr(xp.linalg)
        assert "TensorFlowArrayApiLinalg" in r


class TestNamespaceSupportsRequiredOps:
    def test_our_namespace_passes(self):
        ns = get_namespace()
        assert namespace_supports_required_ops(ns) is True


class TestNamespaceSupportsRequiredOpsFake:
    def test_incomplete_namespace_returns_false(self):
        class FakeNs:
            pass

        assert namespace_supports_required_ops(FakeNs()) is False

    def test_namespace_missing_one_op(self):
        class PartialNs:
            def asarray(self, x, copy=None):
                return x

        assert namespace_supports_required_ops(PartialNs()) is False

    def test_none_returns_false(self):
        assert namespace_supports_required_ops(None) is False
