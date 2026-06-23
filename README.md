# array-api-compat-tf

[![CI](https://github.com/aam-at/array-api-compat-tf/actions/workflows/ci.yml/badge.svg)](https://github.com/aam-at/array-api-compat-tf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This is a small compatibility layer that patches
[`tensorflow.experimental.numpy`](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy)
so TensorFlow tensors can be used with libraries written against the
[Array API standard](https://data-apis.org/array-api/latest/). It also serves as
a drop-in enhancement for
[`array-api-compat`](https://github.com/data-apis/array-api-compat): call
`array_namespace()` with PyTorch, JAX, NumPy, or TensorFlow arrays and get back
the correct namespace.

Platform support follows TensorFlow and the optional backends you install.
See the documentation for more details. Build it locally with
`pixi run -e docs docs`, or read the source in [`docs/index.md`](docs/index.md).

## Quick install

From a git checkout or as a git dependency in uv, install the package plus the
backends you need. Groups are named `tensorflow`, `numpy`, `pytorch`, `jax`,
`tensorflow-gpu`, and `dev`.

**uv project from git** (use extras on the dependency):

```bash
uv add "array-api-compat-tf[tensorflow] @ git+https://github.com/aam-at/array-api-compat-tf"
```

Or in `pyproject.toml`:

```toml
dependencies = ["array-api-compat-tf[tensorflow]"]

[tool.uv.sources]
array-api-compat-tf = { git = "https://github.com/aam-at/array-api-compat-tf" }
```

**Git checkout** (use dependency groups):

```bash
pip install .
pip install --group tensorflow
pip install --group dev
```

Optional NVIDIA CUDA TensorFlow on Linux:

```bash
pip install --group tensorflow-gpu
```

## Related projects

- [Python Array API standard](https://data-apis.org/array-api/latest/)
- [array-api-compat](https://github.com/data-apis/array-api-compat) — NumPy, PyTorch, JAX, and other backend shims
- [TensorFlow `tf.experimental.numpy`](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy)

## License

MIT — see [LICENSE](LICENSE).
