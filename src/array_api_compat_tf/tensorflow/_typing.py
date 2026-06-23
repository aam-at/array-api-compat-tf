"""Type aliases for the TensorFlow backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["Array", "ArrayLike", "Device", "DType", "Shape"]

try:
    import tensorflow as tf

    Array = tf.Tensor
    Device = str
    DType = tf.DType
except Exception:  # pragma: no cover - tensorflow may be unavailable
    Array = Any  # type: ignore[misc, assignment]
    Device = str
    DType = Any  # type: ignore[misc, assignment]

ArrayLike = Array | Sequence[Any] | int | float | bool
Shape = Sequence[int]
