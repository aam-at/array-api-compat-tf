Helper Functions
================

.. currentmodule:: array_api_compat_tf

In addition to the patched TensorFlow namespace, this package provides helper
functions that mirror those in ``array_api_compat`` and additionally recognise
TensorFlow tensors.

Entry-point Helpers
-------------------

The :func:`array_namespace` function is the primary entry-point for Array API
consuming libraries.

.. autofunction:: array_namespace
.. autofunction:: is_array

TensorFlow Inspection Helpers
-----------------------------

.. autofunction:: is_tensorflow_array
.. autofunction:: is_tensorflow_namespace

Array Method Helpers
--------------------

These helpers mirror ``array_api_compat`` and delegate to it for non-TF arrays.

.. autofunction:: device
.. autofunction:: to_device
.. autofunction:: size

TensorFlow Namespace (advanced)
-------------------------------

.. autofunction:: get_namespace
.. autofunction:: namespace_supports_required_ops
.. autoclass:: TensorFlowArrayApiNamespace
   :members:
.. autoclass:: TensorFlowArrayApiLinalg
   :members:
