"""Tests for cross-backend integration helpers."""

import numpy as np
import pytest
import tensorflow as tf

from array_api_compat_tf import (
    TensorFlowArrayApiNamespace,
    array_namespace,
    device,
    get_namespace,
    is_array,
    is_tensorflow_array,
    is_tensorflow_namespace,
    size,
    to_device,
)
from tests.helpers import assert_array_equal


class TestIsTensorflowArray:
    def test_tensor(self):
        assert is_tensorflow_array(tf.constant([1.0])) is True

    def test_variable(self):
        assert is_tensorflow_array(tf.Variable([1.0])) is True

    def test_numpy(self):
        assert is_tensorflow_array(np.array([1.0])) is False

    def test_python_scalar(self):
        assert is_tensorflow_array(1.0) is False

    def test_none(self):
        assert is_tensorflow_array(None) is False

    def test_list(self):
        assert is_tensorflow_array([1.0, 2.0]) is False


class TestIsTensorflowNamespace:
    def test_our_namespace(self):
        ns = get_namespace()
        assert is_tensorflow_namespace(ns) is True

    def test_numpy_namespace(self):
        assert is_tensorflow_namespace(np) is False

    def test_arbitrary_object(self):
        assert is_tensorflow_namespace("hello") is False


class TestIsArray:
    def test_tf_tensor(self):
        assert is_array(tf.constant([1.0])) is True

    def test_tf_variable(self):
        assert is_array(tf.Variable([1.0])) is True

    def test_numpy_array(self):
        assert is_array(np.array([1.0])) is True

    def test_python_scalar(self):
        assert is_array(1.0) is False

    def test_none(self):
        assert is_array(None) is False


class TestArrayNamespace:
    def test_tf_tensor_returns_patched_ns(self):
        t = tf.constant([1.0])
        ns = array_namespace(t)
        assert isinstance(ns, TensorFlowArrayApiNamespace)

    def test_tf_variable_returns_patched_ns(self):
        v = tf.Variable([1.0])
        ns = array_namespace(v)
        assert isinstance(ns, TensorFlowArrayApiNamespace)

    def test_numpy_array_returns_numpy_compat(self):
        a = np.array([1.0])
        ns = array_namespace(a)
        assert not isinstance(ns, TensorFlowArrayApiNamespace)

    def test_multiple_tf_tensors(self):
        a = tf.constant([1.0])
        b = tf.constant([2.0])
        ns = array_namespace(a, b)
        assert isinstance(ns, TensorFlowArrayApiNamespace)

    def test_cached_same_instance(self):
        a = tf.constant([1.0])
        ns1 = array_namespace(a)
        ns2 = array_namespace(a)
        assert ns1 is ns2

    def test_roundtrip_ops(self):
        t = tf.constant([1.0, -2.0, 3.0])
        ns = array_namespace(t)
        out = ns.clip(t, min=0.0)
        assert_array_equal(out, [1.0, 0.0, 3.0])


@pytest.mark.torch
class TestArrayNamespaceTorch:
    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_torch_tensor(self):
        import torch

        t = torch.tensor([1.0, 2.0])
        ns = array_namespace(t)
        assert not isinstance(ns, TensorFlowArrayApiNamespace)
        out = ns.sum(t)
        assert float(out) == 3.0


class TestDevice:
    def test_tf_tensor(self):
        t = tf.constant([1.0])
        d = device(t)
        assert isinstance(d, str)

    def test_numpy_array(self):
        a = np.array([1.0])
        d = device(a)
        assert d is not None


class TestToDevice:
    def test_tf_tensor(self):
        t = tf.constant([1.0])
        out = to_device(t, "/cpu:0")
        assert isinstance(out, tf.Tensor)
        assert_array_equal(out, [1.0])


class TestSize:
    def test_tf_tensor(self):
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        assert size(t) == 4

    def test_tf_scalar(self):
        t = tf.constant(1.0)
        assert size(t) == 1

    def test_numpy_array(self):
        a = np.array([1.0, 2.0, 3.0])
        assert size(a) == 3
