"""Array API compatibility layer for TensorFlow.

Patches ``tensorflow.experimental.numpy`` to provide a namespace that
conforms to the Python Array API standard, and extends
:mod:`array_api_compat`'s :func:`array_namespace` to TensorFlow tensors. Other
arrays get :mod:`array_api_compat`'s namespaces unchanged: importing this
package modifies no other module.
"""

__version__ = "0.1.0"

from array_api_compat_tf.common import (
    ArrayApiNamespace,
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
    "ArrayApiNamespace",
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
