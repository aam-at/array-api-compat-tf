"""Drop-in enhancement for :mod:`array_api_compat` with TensorFlow support.

Provides :func:`array_namespace`, :func:`is_array`,
:func:`is_tensorflow_array`, and :func:`is_tensorflow_namespace` that mirror
the helpers in ``array_api_compat`` (e.g. ``is_torch_array``) but also
recognise TensorFlow tensors and return a patched Array API namespace for
them.

Usage::

    from array_api_compat_tf import array_namespace, is_array

    xp = array_namespace(some_tensor)   # works for torch, jax, numpy AND tensorflow
    assert is_array(some_tensor)

The detection logic avoids importing TensorFlow eagerly so that these
helpers remain cheap to call when TF is not installed or not yet imported.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, TypeVar

import array_api_compat

from array_api_compat_tf.tensorflow._namespace import (
    TensorFlowArrayApiNamespace,
)
from array_api_compat_tf.tensorflow._namespace import (
    get_namespace as _get_tf_namespace,
)

__all__ = [
    "array_namespace",
    "device",
    "is_array",
    "is_tensorflow_array",
    "is_tensorflow_namespace",
    "size",
    "to_device",
]

# Type alias for any Array API-compatible namespace.
ArrayApiNamespace = TensorFlowArrayApiNamespace | ModuleType

_T = TypeVar("_T")

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None

# ---------------------------------------------------------------------------
# Fast subclass check (mirrors array_api_compat._helpers._issubclass_fast)
# ---------------------------------------------------------------------------
# Uses ``sys.modules`` to look up classes by module + name *without* importing
# the module.  This means ``is_tensorflow_array()`` is near-zero-cost when
# TensorFlow has not been imported yet.  The result is cached per (cls,
# modname, clsname) triple.


# ---------------------------------------------------------------------------
# Type-checking helpers
# ---------------------------------------------------------------------------


def is_tensorflow_array(x: object) -> bool:
    """Return ``True`` if *x* is a TensorFlow ``Tensor`` or ``Variable``.

    This function does **not** import TensorFlow if it has not already
    been imported and is therefore cheap to use.

    Implementation note: uses ``sys.modules`` + ``isinstance`` rather than
    ``type(x).__module__.startswith("tensorflow")`` so the function is safe
    to call inside ``torch.compile(fullgraph=True)``.  Dynamo cannot trace
    ``str.startswith`` on a dynamically-resolved ``__module__`` attribute
    (it raises ``Unsupported method call``), but it handles ``isinstance``
    and constant-key ``sys.modules`` lookups correctly.
    """
    import sys

    tf = sys.modules.get("tensorflow")
    if tf is None:
        return False
    return isinstance(x, (tf.Tensor, tf.Variable))


def is_tensorflow_namespace(xp: object) -> bool:
    """Return ``True`` if *xp* is a :class:`TensorFlowArrayApiNamespace`."""
    return isinstance(xp, TensorFlowArrayApiNamespace)


def is_array(x: object) -> bool:
    """Return ``True`` if *x* is a recognised array (TF, torch, numpy, …)."""
    return is_tensorflow_array(x) or array_api_compat.is_array_api_obj(x)


# ---------------------------------------------------------------------------
# Namespace resolution
# ---------------------------------------------------------------------------


def array_namespace(
    *arrays: object,
    api_version: str | None = None,
    use_compat: bool | None = None,
) -> ArrayApiNamespace:
    """Return an Array API namespace for the given *arrays*.

    Drop-in replacement for :func:`array_api_compat.array_namespace` that
    additionally handles TensorFlow tensors.  For non-TF arrays the call is
    forwarded to ``array_api_compat`` unchanged.

    Parameters
    ----------
    *arrays:
        One or more arrays whose backend determines the namespace.
    api_version:
        Passed through to ``array_api_compat.array_namespace``.
    use_compat:
        Passed through to ``array_api_compat.array_namespace``.

    Returns
    -------
    namespace
        An Array API-compatible module or :class:`TensorFlowArrayApiNamespace`.
    """
    saw_torch_array = False
    for a in arrays:
        if is_tensorflow_array(a):
            return _get_tf_namespace()
        if torch is not None and isinstance(a, torch.Tensor):
            saw_torch_array = True

    if saw_torch_array:
        if use_compat is False:
            return torch
        if api_version is not None:
            from array_api_compat.common import _helpers as _array_api_helpers

            _array_api_helpers._check_api_version(api_version)
        import array_api_compat.torch as torch_namespace

        return torch_namespace

    # Non-TF path — delegate to array_api_compat
    kwargs: dict[str, Any] = {}
    if api_version is not None:
        kwargs["api_version"] = api_version
    if use_compat is not None:
        kwargs["use_compat"] = use_compat
    return array_api_compat.array_namespace(*arrays, **kwargs)


# ---------------------------------------------------------------------------
# device / to_device / size helpers (mirror array_api_compat)
# ---------------------------------------------------------------------------


def device(x: object, /) -> Any:
    """Return the device of *x*.

    For TF tensors returns the ``x.device`` string; otherwise delegates
    to :func:`array_api_compat.device`.
    """
    if is_tensorflow_array(x):
        x_device = getattr(x, "device", None)  # type: ignore[union-attr]
        return x_device if x_device is not None else "cpu"
    return array_api_compat.device(x)


def to_device(x: object, device: Any, /, *, stream: Any = None) -> Any:
    """Move *x* to *dev*.

    For TF tensors uses ``tf.identity`` on the target device; otherwise
    delegates to :func:`array_api_compat.to_device`.
    """
    if is_tensorflow_array(x):
        import tensorflow as tf

        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        current_device = getattr(x, "device", None)  # type: ignore[union-attr]
        if current_device == device:
            return x
        try:
            with tf.device(device):
                return tf.identity(x)
        except Exception as exc:
            raise ValueError(f"Unsupported device {device!r}") from exc
    return array_api_compat.to_device(x, device, stream=stream)


def size(x: object, /) -> int | None:
    """Return total number of elements, or ``None`` for dynamic shapes."""
    if is_tensorflow_array(x):
        s = x.shape  # type: ignore[union-attr]
        if s.rank is None:
            return None
        total = 1
        for dim in s.as_list():
            if dim is None:
                return None
            total *= dim
        return total
    import math as _math

    return _math.prod(x.shape)  # type: ignore[union-attr]
