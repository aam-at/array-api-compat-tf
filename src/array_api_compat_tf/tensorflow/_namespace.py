"""TensorFlow Array API namespace wrapper."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from array_api_compat_tf.tensorflow import _aliases
from array_api_compat_tf.tensorflow._info import __array_namespace_info__ as _ArrayNamespaceInfo
from array_api_compat_tf.tensorflow.linalg import TensorFlowArrayApiLinalg

if TYPE_CHECKING:
    from types import ModuleType

__all__ = [
    "TensorFlowArrayApiNamespace",
    "get_namespace",
    "namespace_supports_required_ops",
]

_log = logging.getLogger(__name__)

class TensorFlowArrayApiNamespace:
    """Array API-compatible namespace backed by TensorFlow.

    Provides implementations for operations where
    ``tensorflow.experimental.numpy`` either lacks support or has
    an incompatible signature:

    * ``asarray`` — adds ``copy`` and ``device`` keywords
    * ``arange`` — adds ``device`` keyword
    * ``zeros``, ``ones``, ``empty``, ``full`` — add ``device`` keyword
    * ``astype`` — adds ``copy`` keyword, uses ``tf.cast``
    * ``clip`` — accepts keyword-only ``min``/``max``
    * ``concat`` — wraps ``concatenate`` with ``axis`` keyword
    * ``cumulative_sum`` — adds ``include_initial`` support
    * ``argmin`` — accepts ``keepdims`` keyword
    * ``searchsorted`` — wraps ``tf.searchsorted`` with Array API signature
    * ``sort`` — adds ``descending`` and ``stable`` keywords
    * ``argsort`` — adds ``descending`` and ``stable`` keywords
    * ``topk`` — normalized tuple return with ``axis`` support
    * ``approximatetopk`` — uses ``tf.math.approx_max_k`` when available and
      falls back to exact ``topk`` otherwise
    * ``linalg.vector_norm`` — full ord support (1, 2, inf, −inf, p)

    All other attribute lookups are forwarded to
    ``tensorflow.experimental.numpy``.
    """

    __array_namespace_info__ = _ArrayNamespaceInfo

    __slots__ = (
        "_tf",
        "_xp",
        "_attr_cache",
        "linalg",
        "bool",
        "float32",
        "float64",
        "int32",
        "int64",
    )

    _tf: ModuleType
    _xp: ModuleType
    linalg: TensorFlowArrayApiLinalg

    def __init__(self, tf: ModuleType) -> None:
        self._tf = tf
        self._xp = tf.experimental.numpy
        self._attr_cache: dict[str, Any] = {}
        self.linalg = TensorFlowArrayApiLinalg(namespace=self)
        self.bool = tf.bool
        self.float32 = tf.float32
        self.float64 = tf.float64
        self.int32 = tf.int32
        self.int64 = tf.int64

    def __repr__(self) -> str:
        return f"<TensorFlowArrayApiNamespace (tf {self._tf.__version__})>"

    def __getattr__(self, name: str) -> Any:
        try:
            return self._attr_cache[name]
        except KeyError:
            pass
        xp = object.__getattribute__(self, "_xp")
        try:
            val = getattr(xp, name)
        except AttributeError:
            raise NotImplementedError(
                f"'{name}' is not available in tensorflow.experimental.numpy "
                f"and is not patched by TensorFlowArrayApiNamespace. "
                f"Check the Array API specification for the correct function name."
            ) from None
        self._attr_cache[name] = val
        return val


_ALIAS_NAMES = _aliases.__all__

for _name in _ALIAS_NAMES:
    _fn = getattr(_aliases, _name)
    setattr(
        TensorFlowArrayApiNamespace,
        _name,
        (lambda f: lambda self, /, *args, **kwargs: f(self, *args, **kwargs))(_fn),
    )

@lru_cache(maxsize=1)
def get_namespace() -> TensorFlowArrayApiNamespace:
    """Return a cached :class:`TensorFlowArrayApiNamespace`.

    The namespace is created once and then cached for all subsequent calls.

    Returns
    -------
    TensorFlowArrayApiNamespace
        A ready-to-use Array API namespace backed by TensorFlow.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.  The error message includes the
        ``pip install`` command needed to fix the problem.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for array_api_compat_tf. "
            "Install it with: pip install \"array-api-compat-tf[tensorflow]\" "
            "(or pip install --group tensorflow from a checkout)"
        ) from exc
    return TensorFlowArrayApiNamespace(tf)


@lru_cache(maxsize=8)
def namespace_supports_required_ops(namespace: object) -> bool:
    """Probe whether *namespace* already supports the ops we patch.

    Returns ``True`` when every patched operation works correctly,
    meaning the caller does **not** need this library's namespace.
    """
    try:
        import tensorflow as tf
    except ImportError:
        _log.debug("TensorFlow not installed; namespace probe skipped")
        return False

    probe = tf.constant([1.0, 3.0], dtype=tf.float32)

    _checks: list[tuple[str, Any]] = [
        ("asarray(copy=False)", lambda: namespace.asarray(probe, copy=False)),  # type: ignore[union-attr]
        ("asarray(device=)", lambda: namespace.asarray(probe, device="/cpu:0")),  # type: ignore[union-attr]
        ("arange(device=)", lambda: namespace.arange(3, device="/cpu:0")),  # type: ignore[union-attr]
        ("astype(copy=False)", lambda: namespace.astype(probe, tf.float32, copy=False)),  # type: ignore[union-attr]
        ("clip(min=)", lambda: namespace.clip(probe, min=0.0)),  # type: ignore[union-attr]
        ("concat()", lambda: namespace.concat([probe, probe], axis=0)),  # type: ignore[union-attr]
        (
            "cumulative_sum(include_initial=True)",
            lambda: namespace.cumulative_sum(probe, axis=0, include_initial=True),  # type: ignore[union-attr]
        ),
        ("argmin(keepdims=True)", lambda: namespace.argmin(probe, axis=0, keepdims=True)),  # type: ignore[union-attr]
        ("topk(k=1)", lambda: namespace.topk(probe, k=1)),  # type: ignore[union-attr]
        ("approximatetopk(k=1)", lambda: namespace.approximatetopk(probe, k=1)),  # type: ignore[union-attr]
        ("searchsorted(right=True)", lambda: namespace.searchsorted(probe, probe, right=True)),  # type: ignore[union-attr]
        ("linalg.vector_norm(ord=1)", lambda: namespace.linalg.vector_norm(probe, ord=1, axis=0)),  # type: ignore[union-attr]
    ]

    for label, check in _checks:
        try:
            check()
        except (TypeError, AttributeError, NotImplementedError) as exc:
            _log.debug("namespace probe failed on %s: %s", label, exc)
            return False
    return True
