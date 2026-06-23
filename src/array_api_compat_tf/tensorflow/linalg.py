"""Linalg sub-namespace for the TensorFlow Array API compatibility layer.

Provides ``linalg.vector_norm`` with full ``ord`` support (1, 2, inf, −inf,
and arbitrary *p*-norms) and delegates all other attribute lookups to
``tensorflow.experimental.numpy.linalg``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from array_api_compat_tf.tensorflow._typing import Array

if TYPE_CHECKING:
    from array_api_compat_tf.tensorflow._namespace import TensorFlowArrayApiNamespace

__all__ = ["TensorFlowArrayApiLinalg"]


class TensorFlowArrayApiLinalg:
    """``linalg`` sub-namespace backed by TensorFlow.

    Provides :meth:`vector_norm` with full ``ord`` support and delegates
    every other attribute to ``tensorflow.experimental.numpy.linalg``.
    """

    __slots__ = ("_namespace", "_attr_cache")

    _namespace: TensorFlowArrayApiNamespace

    def __init__(self, *, namespace: TensorFlowArrayApiNamespace) -> None:
        self._namespace = namespace
        self._attr_cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        return "<TensorFlowArrayApiLinalg>"

    def vector_norm(
        self,
        x: Array,
        *,
        ord: float | int = 2,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Compute the vector norm of *x*.

        Parameters
        ----------
        x : Array
            Input tensor.
        ord : int or float, optional
            Order of the norm.  Supported values:

            * ``1`` — sum of absolute values (L1 / Manhattan).
            * ``2`` (default) — Euclidean norm.
            * ``math.inf`` — maximum absolute value.
            * ``-math.inf`` — minimum absolute value.
            * Any other positive ``int`` / ``float`` *p* — generalised
              *p*-norm: ``(Σ|xᵢ|^p)^(1/p)``.
        axis : int or None, optional
            Axis along which to compute the norm.  ``None`` computes over
            all elements.
        keepdims : bool, optional
            If ``True``, the reduced axis is retained with size 1.

        Returns
        -------
        Array
            The computed norm.

        Raises
        ------
        NotImplementedError
            If *ord* is ``0`` (counting non-zero elements is not supported)
            or an unrecognised type.

        Examples
        --------
        >>> from array_api_compat_tf import get_namespace
        >>> xp = get_namespace()
        >>> t = xp.asarray([3.0, 4.0])
        >>> float(xp.linalg.vector_norm(t, ord=2, axis=0))
        5.0
        >>> float(xp.linalg.vector_norm(t, ord=1, axis=0))
        7.0
        """
        ns = self._namespace
        abs_x = ns.abs(x)
        if ord == 1:
            return ns.sum(abs_x, axis=axis, keepdims=keepdims)
        if ord == 2:
            return ns.sqrt(ns.sum(abs_x * abs_x, axis=axis, keepdims=keepdims))
        if ord == math.inf:
            return ns.max(abs_x, axis=axis, keepdims=keepdims)
        if ord == -math.inf:
            return ns.min(abs_x, axis=axis, keepdims=keepdims)
        if isinstance(ord, int | float):
            if ord == 0:
                raise NotImplementedError(
                    "vector_norm does not support ord=0 (counting non-zero "
                    "elements). Use ord=1, ord=2, or any positive p-norm."
                )
            # General p-norm: (Σ|xᵢ|^p) ^ (1/p)
            return ns.sum(abs_x**ord, axis=axis, keepdims=keepdims) ** (1.0 / ord)
        raise NotImplementedError(
            f"vector_norm does not support ord={ord!r} "
            f"(type {type(ord).__name__}). Supported values: numeric "
            f"scalars (1, 2, math.inf, -math.inf, or any float p)."
        )

    def __getattr__(self, name: str) -> Any:
        # Fallback: delegate to the underlying
        # ``tensorflow.experimental.numpy.linalg`` sub-module so that
        # operations like ``linalg.inv`` or ``linalg.solve`` work without
        # explicit wrappers.
        try:
            return self._attr_cache[name]
        except KeyError:
            pass
        ns = object.__getattribute__(self, "_namespace")
        linalg = getattr(ns._xp, "linalg", None)
        if linalg is not None:
            try:
                val = getattr(linalg, name)
            except AttributeError:
                pass
            else:
                self._attr_cache[name] = val
                return val
        raise NotImplementedError(
            f"'{name}' is not available in tensorflow.experimental.numpy.linalg "
            f"and is not patched by TensorFlowArrayApiLinalg. "
            f"Check the Array API specification for the correct function name."
        )
