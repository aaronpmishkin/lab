"""
CuPY linear algebra backend.
"""
from typing import Union, Tuple, Any, Dict

import numpy as np
from scipy.special import logsumexp  # type: ignore
from scipy.linalg import solve  # type: ignore


default_dtype: Dict[str, type] = {"value": np.float32}
active_device: Dict[str, str] = {"value": "cpu"}

# scaffolding


def np_set_device(device_name: str):
    """Set the default device for linear algebra operations.
    :param device_name: a string identifying the device to use.
    """
    globals()["active_device"]["value"] = device_name


def np_set_global_dtype(dtype: np.dtype):
    """Set the default device for linear algebra operations.
    :param dtype: a string identifying the dtype to use.
    """
    globals()["default_dtype"]["value"] = dtype


def numpy_toggle_autodiff(use_autodiff: bool):
    """Toggle auto-diff engine.
    :param use_autodiff: whether or not to enable the autodiff engine.
    """

    pass


# functions


def np_safe_divide(x: np.ndarray, y: np.ndarray):
    return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))


def np_transpose(x: np.ndarray, dim0: int, dim1: int) -> np.ndarray:
    dims = np.arange(len(x.shape))
    dims[[dim0, dim1]] = [dim1, dim0]

    return np.transpose(x, dims)


def np_to_scalar(x: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        assert x.size == 1
        return x[0]

    return x


def np_to_np(x: np.ndarray) -> np.ndarray:
    return np.array(x)


def np_zeros(shape: Union[int, Tuple[int, ...]], dtype: type = None) -> np.ndarray:
    dtype = default_dtype["value"] if dtype is None else dtype
    return np.zeros(shape, dtype=dtype)


def np_ones(shape: Union[int, Tuple[int, ...]], dtype: type = None) -> np.ndarray:
    dtype = default_dtype["value"] if dtype is None else dtype
    return np.ones(shape, dtype=dtype)


def np_tensor(x: Any, dtype: type = None) -> np.ndarray:
    if dtype is None:
        dtype = default_dtype["value"]
    return np.array(x, dtype=dtype)


def np_cumsum(x: np.ndarray, axis, reverse: bool = False) -> np.ndarray:
    if reverse:
        return x[..., ::-1].cumsum(axis=-1)[..., ::-1]

    return np.cumsum(x, axis=axis)


np_map = {
    # scaffolding
    "set_device": np_set_device,
    "set_global_dtype": np_set_global_dtype,
    "toggle_autodiff": numpy_toggle_autodiff,
    # functions
    "safe_divide": np_safe_divide,
    "concatenate": np.concatenate,
    "sum": np.sum,
    "mean": np.mean,
    "multiply": np.multiply,
    "divide": np.divide,
    "matmul": np.matmul,
    "zeros": np_zeros,
    "zeros_like": np.zeros_like,
    "ones": np_ones,
    "ones_like": np.ones_like,
    "tensor": np_tensor,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "smax": np.maximum,
    "smin": np.minimum,
    "max": np.max,
    "min": np.min,
    "argmin": np.argmin,
    "argmax": np.argmax,
    "cumsum": np_cumsum,
    "flip": np.flip,
    "diag": np.diag,
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "arange": np.arange,
    "logsumexp": logsumexp,
    "digitize": np.digitize,
    "expand_dims": np.expand_dims,
    "transpose": np_transpose,
    "unique": np.unique,
    "stack": np.stack,
    "allclose": np.allclose,
    "size": np.size,
    "sign": np.sign,
    "where": np.where,
    "all": np.all,
    "any": np.any,
    "eye": np.eye,
    "solve": solve,
    "isnan": np.isnan,
    "floor": np.floor,
    "ceil": np.ceil,
    "to_scalar": np_to_scalar,
    "to_np": np_to_np,
    "ravel": np.ravel,
    "dot": np.dot,
    "copy": np.copy,
    "isin": np.isin,
    "logical_not": np.logical_not,
    "sort": np.sort,
    "squeeze": np.squeeze,
    # constants
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    # variables
    "default_dtype": default_dtype,
    "active_device": active_device,
}
