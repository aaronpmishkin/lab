"""
Linear algebra backends.
"""

from typing import Union, Optional, Dict, cast

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

# ===== Backends ===== #

backends: Dict[str, Backend] = {}


# ===== Public Interface ===== #
def reset():
    """
    Reset LAB package. This will remove all instantiated backends.
    """
    globals()["backends"] = {}


def get_backend(
    name: Optional[str] = None,
    impl: Optional[BackendEnum] = None,
    device: DeviceEnum = "cpu",
    dtype: DtypeEnum = "float32",
    use_autodiff: bool = False,
    seed: int = 650,
    lazy: Optional[bool] = False,
) -> Backend:
    """Create a new or load an existing linear algebra backend.
    :param name: a name for the linear algebra backend. Defaults to `None`, in which case the backend is anonymous.
    :param impl: the implementation to use for the linear algebra backend.
    :param device: the device on which to linear algebra computations. Typically "cpu" or "cuda".
    :param dtype: the default data type to use when creating tensors. Typically "float32" or "float64".
    :param use_autodiff: whether or not to leave reverse mode autodiff active (if supported).
    :param seed: an optional seed for the default numpy random number generator.
    :param lazy: whether or not to create backend without initializing it.
    :returns: instance of Backend
    """
    # attempt to load backend
    backend: Backend

    if name is not None and name in backends:
        backend = globals()["backends"][name]

        # instantiate the lazy backend.
        if isinstance(backend, LazyBackend) and not lazy:
            inner_be = create_backend(name, impl, device, dtype, use_autodiff, seed)
            backend.set_backend(inner_be)

    elif name is not None and lazy:
        lazy_be = LazyBackend()
        backend = cast(Backend, lazy_be)
        globals()["backends"][name] = backend

    elif name is None and lazy:
        raise ValueError("LazyBackend instances cannot be created anonymously!")

    elif name is not None and impl is None:
        raise ValueError(
            f"Backend with name {name} has not been created but 'impl' is None! An implementation identifier must be provided when creating a backend for the first time."
        )

    elif name is None and impl is None:
        raise ValueError(
            "'impl' must be specified when requesting an anonymous backend."
        )
    else:
        backend = create_backend(name, impl, device, dtype, use_autodiff, seed)

        if name is not None:
            globals()["backends"][name] = backend

    return backend


def create_backend(
    name: Optional[str] = None,
    impl: Optional[BackendEnum] = None,
    device: DeviceEnum = "cpu",
    dtype: DtypeEnum = "float32",
    use_autodiff: bool = False,
    seed: int = 650,
) -> Backend:
    """Create a new or load an existing linear algebra backend.
    :param name: a name for the linear algebra backend. Defaults to `None`, in which case the backend is anonymous.
    :param impl: the implementation to use for the linear algebra backend.
    :param device: the device on which to linear algebra computations. Typically "cpu" or "cuda".
    :param dtype: the default data type to use when creating tensors. Typically "float32" or "float64".
    :param use_autodiff: whether or not to leave reverse mode autodiff active (if supported).
    :param seed: an optional seed for the default numpy random number generator.
    :returns: instance of Backend
    """
    backend: Backend

    if impl == NUMPY:
        backend = NumpyBackend(device, dtype, use_autodiff, name, seed)
    elif impl == TORCH:
        backend = TorchBackend(device, dtype, use_autodiff, name, seed)

    else:
        raise ValueError(f"Backend with implementation {impl} not supported!")

    return backend
