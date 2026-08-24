"""Array API inspection namespace for TensorFlow."""

from __future__ import annotations

from typing import Any

__all__ = ["__array_namespace_info__"]


class __array_namespace_info__:
    """Minimal Array API inspection namespace for TensorFlow."""

    __module__ = "tensorflow"

    def capabilities(self) -> dict[str, bool]:
        """Return a dict of boolean flags describing optional Array API support.

        Returns
        -------
        dict[str, bool]
            ``"boolean indexing"`` and ``"data-dependent shapes"``, both
            ``True`` for TensorFlow in eager mode.
        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
        }

    def default_device(self) -> str:
        """Return the default device: TensorFlow's CPU placement string."""
        return "/cpu:0"

    def default_dtypes(self) -> dict[str, Any]:
        """Return the default dtypes used for array creation, by kind."""
        import tensorflow as tf

        return {
            "real floating": tf.float64,
            "complex floating": tf.complex128,
            "integral": tf.int64,
            "indexing": tf.int64,
        }

    def dtypes(self, *, device: str | None = None, kind: str | None = None) -> tuple[Any, ...]:
        """Return the dtypes supported by TensorFlow, optionally filtered by *kind*.

        Parameters
        ----------
        device : str or None, optional
            Ignored — TensorFlow supports the same dtypes on every device
            this namespace exposes.
        kind : str or None, optional
            One of ``"bool"``, ``"signed integer"``, ``"unsigned integer"``,
            ``"integral"``, ``"real floating"``, ``"complex floating"``, or
            ``None`` for all dtypes.

        Returns
        -------
        tuple
            The matching ``tf.DType`` values.

        Raises
        ------
        ValueError
            If *kind* is not a recognised dtype kind.
        """
        import tensorflow as tf

        del device
        all_dtypes = (
            tf.bool,
            tf.int8,
            tf.int16,
            tf.int32,
            tf.int64,
            tf.uint8,
            tf.uint16,
            tf.uint32,
            tf.uint64,
            tf.float16,
            tf.bfloat16,
            tf.float32,
            tf.float64,
            tf.complex64,
            tf.complex128,
        )
        if kind is None:
            return all_dtypes
        if kind == "bool":
            return (tf.bool,)
        if kind in {"signed integer", "unsigned integer", "integral"}:
            return tuple(
                dt for dt in all_dtypes if dt.name.startswith("int") or dt.name.startswith("uint")
            )
        if kind == "real floating":
            return (tf.float16, tf.bfloat16, tf.float32, tf.float64)
        if kind == "complex floating":
            return (tf.complex64, tf.complex128)
        raise ValueError(f"Unsupported dtype kind: {kind!r}")

    def devices(self) -> tuple[str, ...]:
        """Return the devices this namespace can place arrays on."""
        return ("/cpu:0",)
