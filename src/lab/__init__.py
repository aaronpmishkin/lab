"""
Linear algebra backends.
"""
from typing import Union, List, Tuple, Optional, Any, Iterable, Dict
from itertools import product

import numpy as np
import torch
import opt_einsum as oe  # type: ignore

from .numpy_backend import np_map
from .torch_backend import torch_map
from .cupy_backend import cp_map, cp


# ===== types ===== #

# base tensor class
Tensor = Union[np.ndarray, torch.Tensor]
TensorList = Union[List[np.ndarray], List[torch.Tensor]]
TensorType = Union[torch.dtype, np.dtype, type]

# ===== constants ===== #

# backends

TORCH = "torch"
NUMPY = "numpy"
CUPY = "cupy"

BACKENDS = [TORCH, NUMPY]


# devices

CPU = "cpu"
CUDA = "cuda"

# dtypes

FLOAT32: str = "float32"
FLOAT64: str = "float64"
INT32: str = "INT32"
INT64: str = "INT64"

DTYPES: List[str] = [FLOAT32, FLOAT64, INT32, INT64]


# testing

TESTABLE_DTYPES = [FLOAT64]
TESTABLE_BACKENDS = [NUMPY, TORCH]

TEST_DICT = {"backend": TESTABLE_BACKENDS, "dtype": TESTABLE_DTYPES}

TEST_GRID = [
    dict(zip(TEST_DICT.keys(), values)) for values in product(*TEST_DICT.values())
]

# ===== globals ===== #


# active backend
backend: str = NUMPY

# generators
np_rng = np.random.Generator
torch_rng = torch.Generator
dtype_str: str = FLOAT32
default_dtype: Dict[str, TensorType] = {"value": np.float32}


# active device for GPU enabled backends.
active_device: Dict[str, str] = {"value": CPU}


def set_seeds(seed: int):
    """Set default seeds for all algebra backends.
    :param seed: the random seed to use.
    """
    # numpy rng
    globals()["np_rng"] = np.random.default_rng(seed=seed)
    np.random.seed(seed)

    # torch rng
    globals()["torch_rng"] = torch.Generator()
    torch_rng.manual_seed(seed)

    # seed all torch devices.
    torch.manual_seed(seed)


def get_dtype():
    """
    Get the default data type used by the backend.
    """
    return default_dtype["value"]


def get_device():
    """
    Get the default device used by the backend.
    """
    return active_device["value"]


def set_dtype(dtype_name: str = "float32"):
    """Set the default data type for linear algebra operations.
    :param dtype: a string identifying the data type to use.
    """
    globals()["dtype_str"] = dtype_name
    if dtype_name == "float32":
        globals()["default_dtype"]["value"] = float32
    elif dtype_name == "float64":
        globals()["default_dtype"]["value"] = float64
    else:
        raise ValueError(f"dtype {dtype_name} not recognized!")

    set_global_dtype(globals()["default_dtype"]["value"])


def set_backend(backend_name: str):
    """Set the global backend to use for linear algebra operations.
    :param backend_name: a string identifying the backend to use.
    """
    globals()["backend"] = backend_name

    if backend_name == NUMPY:
        globals()["driver"] = np
        for name, gl in np_map.items():
            globals()[name] = gl

    elif backend_name == CUPY:
        globals()["driver"] = cp
        for name, gl in np_map.items():
            globals()[name] = gl

    elif backend_name == TORCH:
        globals()["driver"] = torch

        for name, gl in torch_map.items():
            globals()[name] = gl
    else:
        raise ValueError(f"Backend {backend_name} not recognized!")

    set_dtype(dtype_str)

    # disable autodiff engine.
    toggle_autodiff(False)


# ===== linalg ===== #


def einsum(self, path: str, *args) -> Tensor:
    """Optimized einsum operators.

    Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing
    contraction order for einsum-like expressions. Journal of Open Source Software, 2018, 3(26), 753
    DOI: https://doi.org/10.21105/joss.00753
    """
    return oe.contract(path, *args, backend=backend)


# =============== #

# default to numpy
set_backend(NUMPY)
