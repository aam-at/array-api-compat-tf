"""Type aliases for the TensorFlow backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["Array", "ArrayLike", "Device", "DType", "Shape"]

Device = str  # Same in both branches below; assigned once so mypy treats it as one alias.

try:
    import tensorflow as tf

    Array = tf.Tensor
    DType = tf.DType
except Exception:  # pragma: no cover - tensorflow may be unavailable
    Array = Any
    DType = Any

ArrayLike = Array | Sequence[Any] | int | float | bool
Shape = Sequence[int]
