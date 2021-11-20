"""
Linear algebra backends.
"""
from typing import Union, List, Tuple, Optional, Any, Iterable, Dict

import numpy as np
import torch
import opt_einsum as oe  # type: ignore

from .backend import Backend
from .lazy_backend import LazyBackend
from .numpy_backend import NumpyBackend
from .torch_backend import TorchBackend

from .types import (
    TORCH,
    NUMPY,
    CUPY,
    CPU,
    CUDA,
    FLOAT32,
    FLOAT64,
    TensorType,
    Tensor,
    TensorList,
    BackendEnum,
    DtypeEnum,
    DeviceEnum,
)

LAB = Union[Backend, LazyBackend]

# ===== Backends ===== #

backends: Dict[str, LAB] = {}


# ===== Public Interface ===== #


def get_backend(
    impl: Optional[BackendEnum] = None,
    name: Optional[str] = None,
    device: DeviceEnum = "cpu",
    dtype: DtypeEnum = "float32",
    use_autodiff: bool = False,
    seed: int = 650,
) -> LAB:
    """Create a new or load an existing linear algebra backend.
    :param impl: the implementation to use for the linear algebra backend.
    :param device: the device on which to linear algebra computations. Typically "cpu" or "cuda".
    :param dtype: the default data type to use when creating tensors. Typically "float32" or "float64".
    :param use_autodiff: whether or not to leave reverse mode autodiff active (if supported).
    :param name: a name for the linear algebra backend. Defaults to `None`, in which case the backend is anonymous.
    :param seed: an optional seed for the default numpy random number generator.
    :returns: instance of Backend
    """
    # attempt to load backend
    backend: Backend

    if name is not None and name in backends:
        return globals()["backends"][name]

    elif name is not None and impl is None:
        raise ValueError(
            f"Backend with name {name} has not been created but 'impl' is None! An implementation identifier must be provided when creating a backend for the first time."
        )

    elif impl is None:
        raise ValueError(
            "'impl' must be specified when requesting an anonymous backend."
        )

    elif impl == NUMPY:
        backend = NumpyBackend(device, dtype, use_autodiff, name, seed)
        globals()["backends"][name] = backend

    elif impl == TORCH:
        backend = TorchBackend(device, dtype, use_autodiff, name, seed)
        globals()["backends"][name] = backend

    else:
        raise ValueError(f"Backend with implementation {impl} not supported!")

    return backend
