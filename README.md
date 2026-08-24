# array-api-compat-tf

[![CI](https://github.com/aam-at/array-api-compat-tf/actions/workflows/ci.yml/badge.svg)](https://github.com/aam-at/array-api-compat-tf/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This is a small wrapper that patches
[`tensorflow.experimental.numpy`](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy)
so TensorFlow tensors are compatible with the
[Array API standard](https://data-apis.org/array-api/latest/). It builds on
[`array-api-compat`](https://github.com/data-apis/array-api-compat), which
already supports NumPy, CuPy, PyTorch, Dask, JAX, ndonnx and `sparse`, adding
TensorFlow to the same `array_namespace()` entry point. If you encounter any
issues, please [open an issue](https://github.com/aam-at/array-api-compat-tf/issues).

See the documentation for more details: [`docs/index.md`](docs/index.md)
(build it locally with `pixi run -e docs docs`).

## Quick install

From a git checkout or as a git dependency in uv, install the package plus the
backends you need. Groups are named `tensorflow`, `numpy`, `pytorch`, `jax`,
`tensorflow-gpu`, `jax-gpu`, and `dev`.

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

Optional NVIDIA CUDA 12 JAX on Linux:

```bash
pip install --group jax-gpu
```

## Related projects

- [Python Array API standard](https://data-apis.org/array-api/latest/)
- [array-api-compat](https://github.com/data-apis/array-api-compat) — NumPy, PyTorch, JAX, and other backend shims
- [TensorFlow `tf.experimental.numpy`](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy)

## License

MIT — see [LICENSE](LICENSE).
