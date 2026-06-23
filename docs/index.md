# Array API compatibility for TensorFlow

This is a small compatibility layer that patches
[`tensorflow.experimental.numpy`](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy)
so TensorFlow tensors can be used with libraries written against the
[Array API standard](https://data-apis.org/array-api/latest/). It also serves as
a drop-in enhancement for
[`array-api-compat`](https://github.com/data-apis/array-api-compat): call
{func}`~array_api_compat_tf.array_namespace` with PyTorch, JAX, NumPy, or
TensorFlow arrays and get back the correct namespace.

Like `array-api-compat`, this is a pure-Python polyfill whose platform support
follows TensorFlow itself: the default install works on **Linux, macOS, and
Windows** with CPU TensorFlow. An optional `gpu` extra is available for NVIDIA
CUDA wheels on Linux.

If you encounter any issues, please
[open an issue](https://github.com/aam-at/array-api-compat-tf/issues).

The patched namespace is implemented against the [2024.12
version](https://data-apis.org/array-api/2024.12/) of the standard, matching
`array-api-compat`.

## Installation

Install from GitHub. Backend sets are exposed as
[dependency groups](https://packaging.python.org/en/latest/specifications/pyproject-toml/#dependency-groups)
when installing from a checkout, and as
[optional dependencies](https://packaging.python.org/en/latest/specifications/pyproject-toml/#optional-dependencies)
(extras) when adding the project as a git/URL dependency in uv or pip.

**uv git dependency** (extras):

```
uv add "array-api-compat-tf[tensorflow] @ git+https://github.com/aam-at/array-api-compat-tf"
```

**Git checkout** (dependency groups):

```
python -m pip install .
python -m pip install --group tensorflow
```

Optional backend groups / extras: `numpy`, `pytorch`, `jax`, and `dev`.

```
python -m pip install --group numpy
python -m pip install --group pytorch
python -m pip install --group jax
python -m pip install --group dev
```

Optional NVIDIA CUDA TensorFlow wheels on Linux:

```
python -m pip install --group tensorflow-gpu
```

You can also import the TensorFlow namespace directly:

```py
import array_api_compat_tf.tensorflow as xp
xp = xp.get_namespace()
```

**Requirements:** Python 3.10+, `array-api-compat` 1.14+, and TensorFlow 2.21+
when using the TensorFlow backend.

## Usage

The typical usage is to get the corresponding Array API namespace from input
arrays using {func}`~array_api_compat_tf.array_namespace`, like

```py
import tensorflow as tf
from array_api_compat_tf import array_namespace

def your_function(x):
    xp = array_namespace(x)
    return xp.clip(x, min=0.0) + xp.linalg.vector_norm(x, ord=2, axis=-1)
```

The same entry point works for every supported backend:

```py
import numpy as np
import torch
from array_api_compat_tf import array_namespace, is_array

for arr in (np.array([1.0, 2.0]), torch.tensor([1.0, 2.0]), tf.constant([1.0, 2.0])):
    xp = array_namespace(arr)
    assert is_array(arr)
```

For TensorFlow inputs, {func}`~array_api_compat_tf.array_namespace` returns a
patched namespace that wraps `tf.experimental.numpy` and overrides only the
operations that differ from the Array API. For all other backends, calls are
forwarded unchanged to `array_api_compat`.

## Patched operations

| Operation | What we add |
| --- | --- |
| `asarray`, `zeros`, `ones`, `empty`, `full`, `arange` | `device=` and/or `copy=` keywords |
| `astype`, `clip`, `concat`, `cumulative_sum` | Array API signatures |
| `searchsorted`, `sort`, `argsort`, `topk`, `approximatetopk`, `take` | Array API signatures |
| `where`, `result_type` | Normalised wrappers |
| `linalg.vector_norm` | Full `ord` support |

Everything else delegates to `tf.experimental.numpy` unchanged.

(relationship)=
## Relationship to `array-api-compat`

[`array-api-compat`](https://github.com/data-apis/array-api-compat) already
provides Array API namespaces for NumPy, CuPy, PyTorch, Dask, JAX, ndonnx, and
Sparse. TensorFlow ships `tf.experimental.numpy`, which covers much of the
standard, but several operations are missing or use incompatible signatures.

**array-api-compat-tf** closes those gaps and plugs TensorFlow into the same
`array_namespace()` workflow as the other backends. It depends on
`array-api-compat` and delegates to it for non-TensorFlow arrays.

(scope)=
## Scope

The scope of this library is limited to:

- patching `tf.experimental.numpy` where it differs from the Array API standard,
- providing [helper functions](helper-functions.rst) that mirror
  `array_api_compat` and additionally recognise TensorFlow tensors,
- small PyTorch compatibility patches applied at import time so cross-backend
  test harnesses behave consistently.

Things that are out of scope include:

- reimplementing operations that TensorFlow already exposes correctly,
- functions that have not yet been standardised,
- replacing TensorFlow's array object with a separate wrapper type.

The goal is to keep this package as a minimal polyfill so consuming libraries
can accept TensorFlow tensors today without waiting for upstream changes.

```{toctree}
:titlesonly:
:hidden:

helper-functions.rst
```
