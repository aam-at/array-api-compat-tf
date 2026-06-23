"""Shared pytest fixtures for the array_api_compat_tf test suite."""

from __future__ import annotations

import pytest

from array_api_compat_tf import get_namespace
from tests.helpers import ATOL, RTOL


def pytest_addoption(parser):
    parser.addoption("--ATOL", action="store", default=ATOL, type=float, help="Absolute tolerance")
    parser.addoption("--RTOL", action="store", default=RTOL, type=float, help="Relative tolerance")


@pytest.fixture(scope="module")
def atol(request):
    return request.config.getoption("--ATOL")


@pytest.fixture(scope="module")
def rtol(request):
    return request.config.getoption("--RTOL")


@pytest.fixture
def xp():
    return get_namespace()


@pytest.fixture(params=["torch", "jax"], ids=["tf-vs-torch", "tf-vs-jax"])
def peer_backend(request):
    """Return ``(name, make_tensor)`` for the comparison backend."""
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        return "torch", lambda arr: torch.tensor(arr)
    jax = pytest.importorskip("jax")
    return "jax", lambda arr: jax.numpy.array(arr)
