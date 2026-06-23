"""Import-time backend patches layered on top of ``array_api_compat``.

This module is imported automatically by ``array_api_compat_tf.__init__`` and
patches non-TensorFlow namespaces so that callers using
``array_api_compat_tf.array_namespace(...)`` see the same extended API across
TensorFlow, PyTorch, JAX, and NumPy.

Current patches:

- ``array_api_compat.torch.clip``: Dynamo-safe scalar checks
- ``array_api_compat.torch.cumulative_sum``: torch-native implementation for
  ``torch.compile(fullgraph=True)``
- ``topk`` / ``approximatetopk``: normalized tuple-returning helpers for
  ``array_api_compat.torch``, ``array_api_compat.numpy``, and JAX namespaces
"""

from __future__ import annotations

import math
import operator
from collections.abc import Hashable
from functools import lru_cache
from typing import Any


def _append_to_all(module: object, *names: str) -> None:
    all_names = getattr(module, "__all__", None)
    if all_names is None:
        return
    updated = list(all_names)
    for name in names:
        if name not in updated:
            updated.append(name)
    setattr(module, "__all__", updated)


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("topk() requires an array with at least one dimension")
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis if axis >= 0 else axis + ndim


def _validate_topk_inputs(x: Any, k: int, axis: int) -> tuple[int, int]:
    normalized_axis = _normalize_axis(axis, x.ndim)
    k = operator.index(k)
    if k < 0:
        raise ValueError("k must be non-negative")

    axis_size = x.shape[normalized_axis]
    if axis_size is not None and k > axis_size:
        raise ValueError(
            f"k must be less than or equal to the size of axis {normalized_axis} "
            f"({axis_size}), got {k}"
        )
    return k, normalized_axis


def _slice_first_k(x: Any, *, axis: int, k: int) -> Any:
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(0, k)
    return x[tuple(slices)]


def _topk_via_argsort(
    x: Any,
    /,
    *,
    k: int,
    axis: int,
    sorted: bool,
    argsort_fn: Any,
    take_along_axis_fn: Any,
) -> tuple[Any, Any]:
    del sorted  # Fallback returns descending order regardless of this hint.
    k, axis = _validate_topk_inputs(x, k, axis)
    indices = argsort_fn(x, axis=axis, descending=True, stable=True)
    indices = _slice_first_k(indices, axis=axis, k=k)
    values = take_along_axis_fn(x, indices, axis=axis)
    return values, indices


def _apply_array_api_compat_clip_patch() -> None:
    try:
        import array_api_compat.torch as _act
        import array_api_compat.torch._aliases as _aliases
    except ImportError:
        return

    import torch

    def _clip_dynamo_safe(x, /, min=None, max=None, **kwargs):
        def _isscalar(a):
            return isinstance(a, (int, float)) or a is None

        if not x.is_floating_point():
            if type(min) is int and min <= torch.iinfo(x.dtype).min:
                min = None
            if type(max) is int and max >= torch.iinfo(x.dtype).max:
                max = None

        if min is None and max is None:
            return torch.clone(x)

        min_is_scalar = _isscalar(min)
        max_is_scalar = _isscalar(max)

        if min_is_scalar and max_is_scalar:
            if (min is not None and math.isnan(min)) or (max is not None and math.isnan(max)):
                return torch.full_like(x, fill_value=torch.nan)
            return torch.clamp(x, min, max, **kwargs)

        a_min = min
        if min is not None and min_is_scalar:
            a_min = torch.as_tensor(min, dtype=x.dtype, device=x.device)

        a_max = max
        if max is not None and max_is_scalar:
            a_max = torch.as_tensor(max, dtype=x.dtype, device=x.device)

        return torch.clamp(x, a_min, a_max, **kwargs)

    _aliases.clip = _clip_dynamo_safe
    _act.clip = _clip_dynamo_safe


def _apply_array_api_compat_cumulative_sum_patch() -> None:
    try:
        import array_api_compat.torch as _act
        import array_api_compat.torch._aliases as _aliases
    except ImportError:
        return

    import torch
    from array_api_compat._internal import get_xp

    def _cumulative_sum_dynamo_safe(
        x, /, xp, *, axis=None, dtype=None, include_initial=False, **kwargs
    ):
        if axis is None:
            if x.ndim > 1:
                raise ValueError(
                    "axis must be specified in cumulative_sum for more than one dimension"
                )
            axis = 0

        res = torch.cumsum(x, dim=axis, dtype=dtype)

        if include_initial:
            initial_shape = list(x.shape)
            initial_shape[axis] = 1
            res = torch.cat(
                [torch.zeros(initial_shape, dtype=res.dtype, device=res.device), res],
                dim=axis,
            )
        return res

    _aliases.cumulative_sum = _cumulative_sum_dynamo_safe
    _act.cumulative_sum = get_xp(torch)(_cumulative_sum_dynamo_safe)


def _apply_array_api_compat_cls_to_namespace_patch() -> None:
    try:
        import array_api_compat.common._helpers as _helpers
    except ImportError:
        return

    @lru_cache(100)
    def _cls_to_namespace_safe(
        cls: type,
        api_version: str | None,
        use_compat: bool | None,
    ) -> tuple[object | None, object | None]:
        if use_compat not in (None, True, False):
            raise ValueError("use_compat must be None, True, or False")
        use_compat_default = use_compat in (None, True)
        cls_ = cls if isinstance(cls, Hashable) else type(cls)

        if _helpers._issubclass_fast(cls_, "numpy", "ndarray") or _helpers._issubclass_fast(
            cls_, "numpy", "generic"
        ):
            if use_compat is True:
                _helpers._check_api_version(api_version)
                from array_api_compat import numpy as xp
            elif use_compat is False:
                import numpy as xp
            else:
                from array_api_compat import numpy as xp
            return xp, _helpers._ClsToXPInfo.MAYBE_JAX_ZERO_GRADIENT

        if issubclass(cls, (int, float, complex, type(None))):
            return None, _helpers._ClsToXPInfo.SCALAR

        if _helpers._issubclass_fast(cls_, "cupy", "ndarray"):
            if use_compat_default:
                _helpers._check_api_version(api_version)
                from array_api_compat import cupy as xp
            else:
                import cupy as xp
            return xp, None

        if _helpers._issubclass_fast(cls_, "torch", "Tensor"):
            if use_compat_default:
                _helpers._check_api_version(api_version)
                from array_api_compat import torch as xp
            else:
                import torch as xp
            return xp, None

        if _helpers._issubclass_fast(cls_, "dask.array", "Array"):
            if use_compat_default:
                _helpers._check_api_version(api_version)
                from array_api_compat.dask import array as xp
            else:
                import dask.array as xp
            return xp, None

        if _helpers._issubclass_fast(cls_, "jax", "Array"):
            return _helpers._jax_namespace(api_version, use_compat), None

        return None, None

    _helpers._cls_to_namespace = _cls_to_namespace_safe


def _apply_array_api_compat_torch_topk_patches() -> None:
    try:
        import array_api_compat.torch as _act
        import array_api_compat.torch._aliases as _aliases
    except ImportError:
        return

    import torch

    def _torch_topk(x, /, *, k, axis=-1, sorted=True):
        k, axis = _validate_topk_inputs(x, k, axis)
        values, indices = torch.topk(x, k, dim=axis, sorted=sorted)
        return values, indices

    def _torch_approximatetopk(x, /, *, k, axis=-1, sorted=True):
        return _torch_topk(x, k=k, axis=axis, sorted=sorted)

    _aliases.topk = _torch_topk
    _act.topk = _torch_topk
    _aliases.approximatetopk = _torch_approximatetopk
    _act.approximatetopk = _torch_approximatetopk
    _append_to_all(_aliases, "topk", "approximatetopk")
    _append_to_all(_act, "topk", "approximatetopk")


def _apply_array_api_compat_numpy_topk_patches() -> None:
    try:
        import array_api_compat.numpy as _acn
        import array_api_compat.numpy._aliases as _aliases
    except ImportError:
        return

    def _numpy_topk(x, /, *, k, axis=-1, sorted=True):
        return _topk_via_argsort(
            x,
            k=k,
            axis=axis,
            sorted=sorted,
            argsort_fn=_aliases.argsort,
            take_along_axis_fn=_aliases.take_along_axis,
        )

    _aliases.topk = _numpy_topk
    _acn.topk = _numpy_topk
    _aliases.approximatetopk = _numpy_topk
    _acn.approximatetopk = _numpy_topk
    _append_to_all(_aliases, "topk", "approximatetopk")
    _append_to_all(_acn, "topk", "approximatetopk")


def _apply_jax_topk_patches() -> None:
    try:
        import jax.lax as lax
        import jax.numpy as jnp
    except ImportError:
        return

    def _jax_argsort(x, *, axis, descending, stable):
        try:
            return jnp.argsort(x, axis=axis, descending=descending, stable=stable)
        except TypeError:
            kwargs = {"stable": stable}
            if not descending:
                return jnp.argsort(x, axis=axis, **kwargs)
            flipped = jnp.flip(jnp.argsort(jnp.flip(x, axis=axis), axis=axis, **kwargs), axis=axis)
            return (x.shape[axis] - 1) - flipped

    # Check once at patch time; avoids per-call hasattr + try/except overhead.
    _has_approx_max_k: bool = hasattr(lax, "approx_max_k")

    def _jax_topk(x, /, *, k, axis=-1, sorted=True):
        del sorted  # JAX top_k always returns values in descending order.
        k, axis = _validate_topk_inputs(x, k, axis)
        # For the last axis use lax.top_k(x, k) without the axis kwarg:
        # compatible with all JAX versions and avoids unnecessary dispatch.
        if axis == x.ndim - 1:
            return lax.top_k(x, k)
        return lax.top_k(x, k, axis=axis)

    def _jax_approximatetopk(x, /, *, k, axis=-1, sorted=True):
        del sorted  # No native unsorted mode.
        k, axis = _validate_topk_inputs(x, k, axis)
        # _has_approx_max_k checked once at patch time to skip try/except when absent.
        if _has_approx_max_k and jnp.issubdtype(x.dtype, jnp.floating):
            try:
                return lax.approx_max_k(x, k, reduction_dimension=axis)
            except Exception:
                pass
        return _jax_topk(x, k=k, axis=axis, sorted=True)

    targets = [jnp]
    try:
        namespace = jnp.empty(0).__array_namespace__()
    except Exception:
        namespace = None
    if namespace is not None and namespace not in targets:
        targets.append(namespace)

    for target in targets:
        setattr(target, "topk", _jax_topk)
        setattr(target, "approximatetopk", _jax_approximatetopk)
        _append_to_all(target, "topk", "approximatetopk")


_apply_array_api_compat_clip_patch()
_apply_array_api_compat_cumulative_sum_patch()
_apply_array_api_compat_cls_to_namespace_patch()
_apply_array_api_compat_torch_topk_patches()
_apply_array_api_compat_numpy_topk_patches()
_apply_jax_topk_patches()
