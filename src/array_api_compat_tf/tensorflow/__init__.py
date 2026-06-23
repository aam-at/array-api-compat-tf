"""TensorFlow Array API compatibility namespace."""

from __future__ import annotations

from typing import Final

from ._info import __array_namespace_info__
from ._namespace import (
    TensorFlowArrayApiNamespace,
    get_namespace,
    namespace_supports_required_ops,
)
from .linalg import TensorFlowArrayApiLinalg

__array_api_version__: Final = "2024.12"

__all__ = [
    "TensorFlowArrayApiLinalg",
    "TensorFlowArrayApiNamespace",
    "__array_api_version__",
    "__array_namespace_info__",
    "get_namespace",
    "namespace_supports_required_ops",
]
