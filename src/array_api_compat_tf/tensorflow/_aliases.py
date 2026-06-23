"""Array API aliases for the TensorFlow backend."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any, Literal

from array_api_compat_tf.tensorflow._typing import Array, ArrayLike, DType, Shape

__all__ = [
    "argmin", "approximatetopk", "arange", "argsort", "asarray", "astype",
    "clip", "concat", "cumulative_sum", "empty", "full", "ones",
    "result_type", "searchsorted", "sort", "take", "topk", "where", "zeros",
]
def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("topk() requires an array with at least one dimension")
    if axis < -ndim or axis >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis if axis >= 0 else axis + ndim


def _build_moveaxis_perm(ndim: int, source: int, destination: int) -> list[int]:
    source = _normalize_axis(source, ndim)
    destination = _normalize_axis(destination, ndim)
    perm = [i for i in range(ndim) if i != source]
    perm.insert(destination, source)
    return perm

def asarray(

    ns,
    x: ArrayLike,
    /,
    *,
    dtype: DType | None = None,
    device: Any | None = None,
    copy: bool | None = None,
) -> Array:
    """Convert *x* to a TensorFlow tensor.

    Parameters
    ----------
    x : array_like
        Input data — a TensorFlow tensor, NumPy array, Python scalar, or
        nested sequence.
    dtype : tf.DType, optional
        Desired element type of the output tensor.
    device : optional
        Device on which to place the result (e.g. ``"/cpu:0"``).  If
        ``None``, the default device is used.
    copy : bool or None, optional
        If ``None`` (default), a copy is made only when necessary.  If
        ``True``, a copy is always made.  If ``False``, a copy is never
        made (the underlying TF implementation may still copy).

    Returns
    -------
    Array
        The input converted to a TensorFlow tensor.

    Raises
    ------
    ValueError
        If *device* is not ``None`` and is not supported.

    Examples
    --------
    >>> xp = get_namespace()
    >>> xp.asarray([1.0, 2.0, 3.0])
    <Array: shape=(3,), dtype=float32, ...>
    >>> xp.asarray([1, 2], dtype=tf.float64, copy=True)
    <Array: shape=(2,), dtype=float64, ...>
    >>> xp.asarray([1.0], device="/cpu:0")
    <Array: shape=(1,), dtype=float32, ...>
    """
    if device is None:
        if copy is None:
            return ns._xp.asarray(x, dtype=dtype)
        return ns._xp.array(x, dtype=dtype, copy=copy)
    try:
        with ns._tf.device(device):
            if copy is None:
                out = ns._xp.asarray(x, dtype=dtype)
            else:
                out = ns._xp.array(x, dtype=dtype, copy=copy)
        return out
    except Exception as exc:
        raise ValueError(f"Unsupported device {device!r}") from exc


def zeros(

    ns,
    shape: Shape | int,
    /,
    *,
    dtype: DType | None = None,
    device: Any | None = None,
) -> Array:
    """Return a tensor of zeros with optional explicit device placement."""
    if device is None:
        return ns._xp.zeros(shape, dtype=dtype)
    try:
        with ns._tf.device(device):
            return ns._xp.zeros(shape, dtype=dtype)
    except Exception as exc:
        raise ValueError(f"Unsupported device {device!r}") from exc


def arange(

    ns,
    start: ArrayLike,
    /,
    stop: ArrayLike | None = None,
    step: ArrayLike = 1,
    *,
    dtype: DType | None = None,
    device: Any | None = None,
) -> Array:
    """Return evenly spaced values within an interval, with optional device
    placement."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    try:
        if device is not None:
            with ns._tf.device(device):
                if stop is None:
                    return ns._xp.arange(start, step=step, **kwargs)
                return ns._xp.arange(start, stop, step=step, **kwargs)
        if stop is None:
            return ns._xp.arange(start, step=step, **kwargs)
        return ns._xp.arange(start, stop, step=step, **kwargs)
    except Exception as exc:
        if device is not None:
            raise ValueError(f"Unsupported device {device!r}") from exc
        raise


def ones(

    ns,
    shape: Shape | int,
    /,
    *,
    dtype: DType | None = None,
    device: Any | None = None,
) -> Array:
    """Return a tensor of ones with optional explicit device placement."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    if device is None:
        return ns._xp.ones(shape, **kwargs)
    try:
        with ns._tf.device(device):
            return ns._xp.ones(shape, **kwargs)
    except Exception as exc:
        raise ValueError(f"Unsupported device {device!r}") from exc


def empty(

    ns,
    shape: Shape | int,
    /,
    *,
    dtype: DType | None = None,
    device: Any | None = None,
) -> Array:
    """Return an uninitialized tensor with optional explicit device
    placement."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    if device is None:
        return ns._xp.empty(shape, **kwargs)
    try:
        with ns._tf.device(device):
            return ns._xp.empty(shape, **kwargs)
    except Exception as exc:
        raise ValueError(f"Unsupported device {device!r}") from exc


def full(

    ns,
    shape: Shape | int,
    /,
    fill_value: ArrayLike,
    *,
    dtype: DType | None = None,
    device: Any | None = None,
) -> Array:
    """Return a tensor filled with *fill_value* and optional device
    placement."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    if device is None:
        return ns._xp.full(shape, fill_value, **kwargs)
    try:
        with ns._tf.device(device):
            return ns._xp.full(shape, fill_value, **kwargs)
    except Exception as exc:
        raise ValueError(f"Unsupported device {device!r}") from exc


def astype(ns, x: Array, dtype: DType, /, *, copy: bool = True) -> Array:

    """Cast *x* to the given *dtype*.

    Parameters
    ----------
    x : Array
        Input tensor.
    dtype : tf.DType
        Target element type (e.g. ``tf.float32``, ``tf.int64``).
    copy : bool, optional
        If ``False`` **and** *x* already has the requested *dtype*, return
        *x* unchanged (no copy).  Defaults to ``True`` (always cast).

    Returns
    -------
    Array
        Tensor with the requested *dtype*.

    Examples
    --------
    >>> xp = get_namespace()
    >>> t = xp.asarray([1.0, 2.0])
    >>> xp.astype(t, tf.float64).dtype
    tf.float64
    >>> xp.astype(t, tf.float32, copy=False) is t  # no-op when same dtype
    True
    """
    if not copy and getattr(x, "dtype", None) == dtype:
        return x
    return ns._tf.cast(x, dtype)


def clip(

    ns,
    x: Array,
    /,
    min: Array | int | float | None = None,
    max: Array | int | float | None = None,
) -> Array:
    """Clamp each element of *x* to the ``[min, max]`` range.

    At least one of *min* or *max* must be provided.

    Parameters
    ----------
    x : Array
        Input tensor.
    min : scalar or Array, optional
        Lower bound.  Elements below this value are set to *min*.
    max : scalar or Array, optional
        Upper bound.  Elements above this value are set to *max*.

    Returns
    -------
    Array
        Clamped tensor with the same shape and dtype as *x*.

    Raises
    ------
    TypeError
        If neither *min* nor *max* is supplied.

    Examples
    --------
    >>> xp = get_namespace()
    >>> t = xp.asarray([-2.0, 0.0, 2.0])
    >>> xp.clip(t, min=0.0)
    <Array: shape=(3,), dtype=float32, numpy=array([0., 0., 2.], ...)>
    >>> xp.clip(t, min=-1.0, max=1.0)
    <Array: shape=(3,), dtype=float32, numpy=array([-1., 0., 1.], ...)>
    """
    if min is None and max is None:
        raise TypeError(
            "clip() requires at least one of 'min' or 'max', but both "
            "were None. Pass min=<value>, max=<value>, or both."
        )
    if min is None:
        return ns._xp.minimum(x, max)
    if max is None:
        return ns._xp.maximum(x, min)
    return ns._xp.clip(x, min, max)


def concat(ns, arrays: Sequence[Array], /, *, axis: int = 0) -> Array:

    """Concatenate a sequence of tensors along an existing axis.

    Parameters
    ----------
    arrays : sequence of Array
        Tensors to join.  All must have the same shape except along
        *axis*.
    axis : int, optional
        Axis along which to concatenate (default ``0``).

    Returns
    -------
    Array
        The concatenated tensor.

    Examples
    --------
    >>> xp = get_namespace()
    >>> a = xp.asarray([1.0, 2.0])
    >>> b = xp.asarray([3.0, 4.0])
    >>> xp.concat([a, b], axis=0)
    <Array: shape=(4,), dtype=float32, numpy=array([1., 2., 3., 4.], ...)>
    """
    return ns._xp.concatenate(arrays, axis=axis)


def cumulative_sum(

    ns,
    x: ArrayLike,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    """Return the cumulative sum of elements along *axis*.

    Parameters
    ----------
    x : Array
        Input tensor.
    axis : int or None, optional
        Axis along which to compute the cumulative sum.  If ``None``,
        the tensor is flattened first and the sum is computed over the
        single resulting axis.
    dtype : tf.DType, optional
        Desired output dtype.  When given, *x* is cast before summation.
    include_initial : bool, optional
        If ``True``, prepend a zero along *axis* so that the output has
        one more element than the input along that axis (matching the
        Array API specification).

    Returns
    -------
    Array
        Cumulative sums.  Shape equals ``x.shape`` unless
        *include_initial* is ``True``, in which case the *axis*
        dimension is incremented by one.

    Examples
    --------
    >>> xp = get_namespace()
    >>> t = xp.asarray([1.0, 2.0, 3.0])
    >>> xp.cumulative_sum(t, axis=0)
    <Array: shape=(3,), dtype=float32, numpy=array([1., 3., 6.], ...)>
    >>> xp.cumulative_sum(t, axis=0, include_initial=True)
    <Array: shape=(4,), dtype=float32, numpy=array([0., 1., 3., 6.], ...)>
    """
    arr = ns.asarray(x)
    if dtype is not None:
        arr = ns.astype(arr, dtype, copy=False)
    if axis is None:
        # Flatten to 1-D when no axis is specified.
        arr = ns._xp.reshape(arr, (-1,))
        axis = 0
    out = ns._xp.cumsum(arr, axis=axis)
    if include_initial:
        # Prepend a zero-slice so the output length is N+1 along *axis*.
        shape = list(out.shape)
        shape[axis] = 1
        zeros = ns.zeros(shape, dtype=out.dtype, device=getattr(out, "device", None))
        out = ns.concat([zeros, out], axis=axis)
    return out


def argmin(

    ns,
    x: ArrayLike,
    /,
    *,
    axis: int | None = None,
    keepdims: bool = False,
) -> Array:
    """Return indices of minimum values, preserving reduced axes when requested."""
    arr = ns.asarray(x)
    if axis is None:
        out = ns._tf.math.argmin(
            ns._xp.reshape(arr, (-1,)), axis=0, output_type=ns._tf.int64
        )
        if keepdims:
            out = ns._tf.reshape(out, ns._tf.ones_like(ns._tf.shape(arr)))
        return out
    out = ns._tf.math.argmin(arr, axis=axis, output_type=ns._tf.int64)
    if keepdims:
        out = ns._tf.expand_dims(out, axis=axis)
    return out


def sort(

    ns,
    x: ArrayLike,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Array:
    """Sort elements along *axis*.

    Array API signature: ``sort(x, /, *, axis=-1, descending=False,
    stable=True)``.

    Parameters
    ----------
    x : Array
        Input tensor.
    axis : int, optional
        Axis along which to sort (default ``-1``).
    descending : bool, optional
        If ``True``, sort in descending order.
    stable : bool, optional
        If ``True`` (default), preserve the relative order of elements
        that compare equal.

    Returns
    -------
    Array
        Sorted tensor with the same shape and dtype as *x*.

    Notes
    -----
    Uses ``tf.sort`` directly (not ``tf.experimental.numpy.sort``)
    because the latter only accepts ``kind="quicksort"`` and rejects
    ``kind="stable"``.  ``tf.sort`` uses a stable sort algorithm by
    default, so the *stable* parameter is honoured implicitly.
    """
    direction = "DESCENDING" if descending else "ASCENDING"
    return ns._tf.sort(x, axis=axis, direction=direction)


def argsort(

    ns,
    x: ArrayLike,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Array:
    """Return indices that sort *x* along *axis*.

    Array API signature: ``argsort(x, /, *, axis=-1, descending=False,
    stable=True)``.

    Parameters
    ----------
    x : Array
        Input tensor.
    axis : int, optional
        Axis along which to sort (default ``-1``).
    descending : bool, optional
        If ``True``, return indices for a descending sort.
    stable : bool, optional
        If ``True`` (default), preserve the relative order of elements
        that compare equal.

    Returns
    -------
    Array
        Integer tensor of indices that sort *x*.

    Notes
    -----
    Uses ``tf.argsort`` directly (not ``tf.experimental.numpy.argsort``)
    because the latter only accepts ``kind="quicksort"`` and rejects
    ``kind="stable"``.  ``tf.argsort`` natively supports a ``stable``
    boolean and a ``direction`` string.
    """
    direction = "DESCENDING" if descending else "ASCENDING"
    return ns._tf.argsort(x, axis=axis, direction=direction, stable=stable)


def _moveaxis(ns, x: Array, source: int, destination: int) -> Array:

    rank = x.shape.rank
    if rank is None:
        raise ValueError("topk() requires tensors with statically known rank")
    if source == destination:
        return x
    return ns._tf.transpose(x, perm=_build_moveaxis_perm(rank, source, destination))


def _take_along_axis(ns, x: Array, indices: Array, /, *, axis: int) -> Array:

    rank = x.shape.rank
    if rank is None:
        raise ValueError("topk() requires tensors with statically known rank")
    axis = _normalize_axis(axis, rank)
    moved_x = _moveaxis(ns, x, axis, rank - 1)
    moved_indices = _moveaxis(ns, indices, axis, rank - 1)
    gathered = ns._tf.gather(
        moved_x,
        moved_indices,
        axis=rank - 1,
        batch_dims=rank - 1,
    )
    return _moveaxis(ns, gathered, rank - 1, axis)


def _slice_first_k(ns, x: Array, /, *, axis: int, k: int) -> Array:

    slices = [slice(None)] * x.shape.rank
    slices[axis] = slice(0, k)
    return x[tuple(slices)]


def _validate_topk_inputs(ns, x: Array, /, *, k: int, axis: int) -> tuple[int, int]:

    rank = x.shape.rank
    if rank is None:
        raise ValueError("topk() requires tensors with statically known rank")
    axis = _normalize_axis(axis, rank)
    k = operator.index(k)
    if k < 0:
        raise ValueError("k must be non-negative")
    axis_size = x.shape[axis]
    if axis_size is not None and k > axis_size:
        raise ValueError(
            f"k must be less than or equal to the size of axis {axis} ({axis_size}), got {k}"
        )
    return k, axis


def _topk_via_argsort(

    ns,
    x: ArrayLike,
    /,
    *,
    k: int,
    axis: int = -1,
    sorted: bool = True,
) -> tuple[Array, Array]:
    del sorted  # Fallback returns descending order regardless of this hint.
    arr = ns.asarray(x)
    k, axis = _validate_topk_inputs(ns, arr, k=k, axis=axis)
    indices = ns.argsort(arr, axis=axis, descending=True, stable=True)
    indices = _slice_first_k(ns, indices, axis=axis, k=k)
    values = _take_along_axis(ns, arr, indices, axis=axis)
    return values, indices


def topk(

    ns,
    x: ArrayLike,
    /,
    *,
    k: int,
    axis: int = -1,
    sorted: bool = True,
) -> tuple[Array, Array]:
    """Return the ``k`` largest values and their indices along *axis*."""
    arr = ns.asarray(x)
    k, axis = _validate_topk_inputs(ns, arr, k=k, axis=axis)
    rank = arr.shape.rank
    if axis == rank - 1:
        values, indices = ns._tf.math.top_k(arr, k=k, sorted=sorted)
        return values, indices

    moved = _moveaxis(ns, arr, axis, rank - 1)
    values, indices = ns._tf.math.top_k(moved, k=k, sorted=sorted)
    values = _moveaxis(ns, values, rank - 1, axis)
    indices = _moveaxis(ns, indices, rank - 1, axis)
    return values, indices


def approximatetopk(

    ns,
    x: ArrayLike,
    /,
    *,
    k: int,
    axis: int = -1,
    sorted: bool = True,
) -> tuple[Array, Array]:
    """Return approximate top-``k`` values and indices, with exact fallback."""
    arr = ns.asarray(x)
    k, axis = _validate_topk_inputs(ns, arr, k=k, axis=axis)
    if arr.dtype.is_floating:
        try:
            rank = arr.shape.rank
            if axis == rank - 1:
                return ns._tf.math.approx_max_k(arr, k)

            moved = _moveaxis(ns, arr, axis, rank - 1)
            values, indices = ns._tf.math.approx_max_k(moved, k)
            values = _moveaxis(ns, values, rank - 1, axis)
            indices = _moveaxis(ns, indices, rank - 1, axis)
            return values, indices
        except Exception:
            pass
    return ns.topk(arr, k=k, axis=axis, sorted=sorted)


def searchsorted(

    ns,
    x1: Array,
    x2: Array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    **kwargs: Any,
) -> Array:
    """Find insertion indices to keep *x1* sorted after inserting *x2*.

    Array API signature: ``searchsorted(x1, x2, /, *, side="left")``.
    For backward compatibility, ``right=True`` is accepted (equivalent to
    ``side="right"``) and ``right=False`` to ``side="left"``.

    Parameters
    ----------
    x1 : Array
        1-D sorted input tensor.
    x2 : Array
        Values to insert.
    side : {'left', 'right'}, optional
        ``'left'`` (default) returns the first suitable index;
        ``'right'`` returns the last.
    **kwargs
        Legacy: ``right=True`` maps to ``side='right'``, ``right=False``
        to ``side='left'``. Any other keyword raises ``TypeError``.

    Returns
    -------
    Array
        Integer tensor of indices with ``dtype=tf.int64``.

    Raises
    ------
    ValueError
        If *side* is not ``'left'`` or ``'right'``.
    TypeError
        If an unexpected keyword argument is passed.

    Examples
    --------
    >>> xp = get_namespace()
    >>> x = xp.asarray([1.0, 3.0, 5.0, 7.0])
    >>> xp.searchsorted(x, xp.asarray([2.0, 6.0]), side="right")
    <Array: shape=(2,), dtype=int64, numpy=array([1, 3])>
    >>> xp.searchsorted(x, xp.asarray([2.0, 6.0]), right=True)  # legacy
    <Array: shape=(2,), dtype=int64, numpy=array([1, 3])>
    """
    if kwargs:
        if set(kwargs) == {"right"}:
            side = "right" if kwargs["right"] else "left"
        else:
            raise TypeError(
                f"searchsorted() got unexpected keyword argument(s): "
                f"{sorted(kwargs)}. Use side='left' or side='right'."
            )
    if side not in {"left", "right"}:
        raise ValueError(
            f"searchsorted 'side' must be 'left' or 'right', "
            f"got {side!r}. Use side='left' for lower-bound indices "
            f"or side='right' for upper-bound indices."
        )
    return ns._tf.searchsorted(x1, x2, side=side, out_type=ns._tf.int64)


def take(ns, x: Array, indices: Array, /, *, axis: int | None = None) -> Array:

    """Select elements from *x* along *axis* (Array API ``take``)."""
    if axis is None:
        x = ns._xp.reshape(x, (-1,))
        axis = 0
    return ns._tf.gather(x, indices, axis=axis)


def where(ns, condition: Array, x1: Array, x2: Array, /) -> Array:

    """Element-wise selection (Array API ``where``)."""
    return ns._tf.where(condition, x1, x2)


def result_type(ns, *arrays_and_dtypes: Any) -> Any:

    """Determine the result dtype from inputs (Array API
    ``result_type``)."""
    import numpy as _np

    dtypes = []
    for a in arrays_and_dtypes:
        if hasattr(a, "dtype"):
            dtypes.append(ns._tf.dtypes.as_dtype(a.dtype))
        else:
            dtypes.append(ns._tf.dtypes.as_dtype(a))
    if len(dtypes) == 1:
        return dtypes[0]
    np_result = _np.result_type(*(d.as_numpy_dtype for d in dtypes))
    return ns._tf.dtypes.as_dtype(np_result)


