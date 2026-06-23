"""Array API compatibility layer for TensorFlow.

Patches ``tensorflow.experimental.numpy`` to provide a namespace that
conforms to the Python Array API standard. Also serves as a drop-in
enhancement for :mod:`array_api_compat` across TensorFlow, PyTorch, JAX,
and NumPy arrays via :func:`array_namespace`.
"""

__version__ = "0.1.0"

from array_api_compat_tf import _pytorch_patches  # noqa: F401  apply third-party patches early
from array_api_compat_tf.common import (
    array_namespace,
    device,
    is_array,
    is_tensorflow_array,
    is_tensorflow_namespace,
    size,
    to_device,
)
from array_api_compat_tf.tensorflow import (
    TensorFlowArrayApiLinalg,
    TensorFlowArrayApiNamespace,
    get_namespace,
    namespace_supports_required_ops,
)

__all__ = [
    "TensorFlowArrayApiLinalg",
    "TensorFlowArrayApiNamespace",
    "array_namespace",
    "device",
    "get_namespace",
    "is_array",
    "is_tensorflow_array",
    "is_tensorflow_namespace",
    "namespace_supports_required_ops",
    "size",
    "to_device",
]
