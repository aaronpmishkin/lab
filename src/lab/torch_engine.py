"""
PyTorch backend for linear algebra operations.
"""
from typing import Union, List, Tuple, Optional, cast, Any, Dict

import torch
import numpy as np

active_device: Dict[str, str] = {"value": "cpu"}
default_dtype: Dict[str, torch.dtype] = {"value": torch.float32}

# scaffolding


def torch_set_device(device_name: str):
    """Set the default device for linear algebra operations.
    :param device_name: a string identifying the device to use.
    """

    if device_name == "cuda":
        # cuda must be available to run in this mode.
        assert torch.cuda.is_available()
        device_name = f"cuda:{torch.cuda.current_device()}"

    globals()["active_device"]["value"] = device_name


def torch_set_global_dtype(dtype: torch.dtype):
    globals()["default_dtype"]["value"] = dtype
    torch.set_default_dtype(dtype)


def torch_toggle_autodiff(use_autodiff: bool):
    """Toggle PyTorch auto-diff engine.
    :param use_autodiff: whether or not to enable the autodiff engine.
    """

    # disable *all* gradient computations.
    if "1.9" in torch.__version__:
        # only available in 1.9.*
        torch.inference_mode(not use_autodiff)
    else:
        torch.set_grad_enabled(use_autodiff)


# functions


def torch_safe_divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Safe divide with pytorch tensors."""
    if not torch.is_tensor(y):
        y = torch.tensor(y)

    return x * torch.nan_to_num(1.0 / y)


def torch_concatenate(*tensors: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
    return torch.cat(*tensors, dim=axis)


def torch_sum(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]] = (),
    keepdims: bool = False,
) -> Union[float, torch.Tensor]:
    return torch.sum(x, dim=axis, keepdim=keepdims)


def torch_mean(
    x: torch.Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]] = (),
    keepdims: bool = False,
) -> Union[float, torch.Tensor]:
    return torch.mean(x, dim=axis, keepdim=keepdims)


def torch_smin(x: torch.Tensor, y: float) -> torch.Tensor:
    """Take the element-wise minimum of a tensor and a scalar."""
    return torch.minimum(x, torch.tensor(y))


def torch_smax(x: torch.Tensor, y: float) -> torch.Tensor:
    """Take the element-wise maximum of a tensor and a scalar."""
    return torch.maximum(x, torch.tensor(y))


def torch_unique(
    x: torch.Tensor, axis: int = None, return_index: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    unique_vals = torch.unique(x, dim=axis)

    if return_index:
        raise NotImplementedError("TODO!")

    return unique_vals


def torch_stack(tensors: List[torch.Tensor], axis=0) -> torch.Tensor:
    return torch.stack(tensors, dim=axis)


def torch_logsumexp(
    x: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> torch.Tensor:
    if axis is None:
        axis = tuple(i for i in range(len(x.shape)))

    return torch.logsumexp(x, dim=axis)


def torch_to_scalar(x: Union[torch.Tensor, float]) -> float:
    if torch.is_tensor(x):
        x = cast(torch.Tensor, x)
        assert torch.numel(x) == 1

        return x.detach().cpu().item()

    x = cast(float, x)
    return x


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# creation ops


def torch_zeros(
    shape: Union[int, Tuple[int, ...]], dtype: torch.dtype = None
) -> torch.Tensor:
    return torch.zeros(shape, device=active_device["value"], dtype=dtype)


def torch_ones(
    shape: Union[int, Tuple[int, ...]], dtype: torch.dtype = None
) -> torch.Tensor:
    return torch.ones(shape, device=active_device["value"], dtype=dtype)


def torch_tensor(x: Any, dtype: torch.dtype = None) -> torch.Tensor:
    return torch.tensor(x, device=active_device["value"], dtype=dtype)


torch_map = {
    # scaffolding
    "set_device": torch_set_device,
    "set_global_dtype": torch_set_global_dtype,
    "toggle_autodiff": torch_toggle_autodiff,
    # functions
    "safe_divide": torch_safe_divide,
    "concatenate": torch_concatenate,
    "sum": torch_sum,
    "mean": torch_mean,
    "multiply": torch.multiply,
    "divide": torch.divide,
    "matmul": torch.matmul,
    "zeros": torch_zeros,
    "zeros_like": torch.zeros_like,
    "ones": torch_ones,
    "ones_like": torch.ones_like,
    "tensor": torch_tensor,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "smax": torch_smax,
    "smin": torch_smin,
    "max": torch.max,
    "min": torch.min,
    "diag": torch.diag,
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "logsumexp": torch_logsumexp,
    "digitize": torch.bucketize,
    "arange": torch.arange,
    "expand_dims": torch.unsqueeze,
    "transpose": torch.transpose,
    "unique": torch_unique,
    "stack": torch_stack,
    "allclose": torch.allclose,
    "size": torch.numel,
    "sign": torch.sign,
    "where": torch.where,
    "all": torch.all,
    "any": torch.any,
    "eye": torch.eye,
    "solve": torch.linalg.solve,
    "isnan": torch.isnan,
    "floor": torch.floor,
    "ceil": torch.ceil,
    "to_scalar": torch_to_scalar,
    "to_np": torch_to_np,
    "ravel": torch.ravel,
    "dot": torch.dot,
    # constants
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    # variables
    "default_dtype": default_dtype,
    "active_device": active_device,
}
