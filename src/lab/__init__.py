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
TensorType = Union[torch.dtype, np.dtype, type]

# ===== constants ===== #

# backends

TORCH = "torch"
NUMPY = "numpy"
CUPY = "cupy"

BACKENDS = [TORCH, NUMPY]

# underlying dtypes
float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64

# devices

CPU: str = "cpu"
CUDA: str = "cuda"

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
np_rng: np.random.Generator = np.random.default_rng()
torch_rng: torch.Generator = torch.random.default_generator
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


def einsum(path: str, *args) -> Tensor:
    """Optimized einsum operators.

    Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing
    contraction order for einsum-like expressions. Journal of Open Source Software, 2018, 3(26), 753
    DOI: https://doi.org/10.21105/joss.00753
    """
    return oe.contract(path, *args, backend=backend)


def all_to_tensor(list_of_x: Iterable[Any], dtype: TensorType = None) -> List[Tensor]:
    return [tensor(x, dtype=dtype) for x in list_of_x]


def all_to_np(list_of_x: Iterable[Tensor]) -> List[np.ndarray]:
    return [to_np(x) for x in list_of_x]


# ===== External Interface ===== #


def toggle_autodiff(use_autodiff: bool):
    """Toggle an auto-diff engine associated with the backend.
    :param use_autodiff: whether or not to enable the autodiff engine.
    """
    pass


def set_global_dtype(dtype: TensorType):
    """Set the default device for linear algebra operations.
    :param device_name: a string identifying the device to use.
    """
    pass


def set_device(device_name: str):
    """Set the default device for linear algebra operations.
    :param device_name: a string identifying the device to use.
    """
    pass


def to_scalar(x: Union[Tensor, float]) -> float:
    """Cast a 1-element tensor into a floating point number.
    :param x: Tensor.
    :returns: scalar value that was stored in x.
    """
    pass


def to_np(x: Tensor) -> np.ndarray:
    """Cast a given tensor into a NumPy array on the CPU.
    :param x: Tensor.
    :returns: np.ndarray(x)
    """
    pass


def safe_divide(x: Tensor, y: Tensor) -> Tensor:
    """Divide two tensors *safely*, where division by 0 is replaced with 0.
    :param x: Tensor.
    :param y: Tensor.
    :returns: x ./ y
    """
    pass


def concatenate(*tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Join sequence of tensors along an exiting axis.
    :param a1, a2, ... : sequence of tensors to join.
    :param axis: the along which to join the tensors.
    :returns: tensors concatenated along the given axis.
    """
    pass


def multiply(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise multiply two tensors with broadcastable shapes.
    :param x: Tensor
    :param y: Tensor
    :returns: x .* y
    """
    pass


def divide(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise divide two tensors with broadcastable shapes.
    Note: this is *not* zero safe. Use 'safe_divide' when 0/0 is possible.
    :param x: Tensor
    :param y: Tensor
    :returns: x ./ y
    """
    pass


def zeros(shape: Union[int, Tuple[int, ...]], dtype: TensorType = None) -> Tensor:
    """Return a tensor of given shape filled with zeros.
    :param shape: the shape of the resulting tensor.
    :param dtype: the data type to use for the tensor.
    :returns: tensor filled with zeros of the desired shape.
    """
    pass


def zeros_like(x: Tensor) -> Tensor:
    """Return a tensor of zeros with the same shape and type as the input tensor.
    :param x: Tensor
    :returns: a tensor with the same shape and type as x, filled with zeros.
    """
    pass


def ones(shape: Union[int, Tuple[int, ...]], dtype: TensorType = None) -> Tensor:
    """Return a tensor of given shape filled with ones.
    :param shape: the shape of the resulting tensor.
    :param dtype: the data type to use for the tensor.
    :returns: tensor filled with ones of the desired shape.
    """
    pass


def ones_like(x: Tensor) -> Tensor:
    """Return a tensor of ones with the same shape and type as the input tensor.
    :param x: Tensor
    :returns: a tensor with the same shape and type as x, filled with ones.
    """
    pass


def matmul(x: Tensor, y: Tensor) -> Tensor:
    """Matrix product of two tensors."""
    pass


def dot(x: Tensor, y: Tensor) -> Tensor:
    """Vector-Vector inner-product."""
    pass


def sum(
    x: Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum of tensor elements over a given axis.
    :param x: tensor over which to sum elements.
    :param axis: (optional) axis or axes along which to perform the sum.
        Supports negative indexing.
    :param keepdims: Set to 'True' if the axis which are reduced should be kept
        with size one in the result.
    :returns: tensor or float.
    """
    pass


def mean(
    x: Tensor,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum of tensor elements over a given axis.
    :param x: tensor over which to take the mean.
    :param axis: (optional) axis or axes along which to perform averaging operation.
        Supports negative indexing.
    :param keepdims: Set to 'True' if the axis which are reduced should be kept
        with size one in the result.
    :returns: tensor or float.
    """
    pass


def tensor(x: Any, dtype: TensorType = None) -> Tensor:
    """Create a new tensor.
    :param x: an array-like object with data for the new tensor.
    :param dtype: the data type to use when constructing the tensor.
    :returns: a new Tensor object with supplied data and type.
    """
    pass


def diag(x: Tensor) -> Tensor:
    """Extract the diagonal of a tensor or construct a diagonal tensor.
    :param x: Tensor. If x is 2-d, then the diagonal of 'x' is extracted.
        If 'x' is 1-d, then 'Diag(x)' is returned.
    :returns: Tensor
    """
    pass


def maximum(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise maximum of the two input tensors.
    :param x: Tensor
    :param y: Tensor
    :returns: max(x, y)
    """
    pass


def minimum(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise minimum of the two input tensors.
    :param x: Tensor
    :param y: Tensor
    :returns: min(x, y)
    """
    pass


def max(x: Tensor) -> Tensor:
    """Element-wise maximum of the input tensor.
    :param x: Tensor
    :returns: max(x)
    """
    pass


def min(x: Tensor) -> Tensor:
    """Element-wise minimum of the input tensor.
    :param x: Tensor
    :returns: max(x)
    """
    pass


def smin(x: Tensor, y: float) -> Tensor:
    """Take the element-wise minimum of a tensor and a scalar.
    :param x: Tensor
    :param y: float
    :returns: min(x, y)
    """
    pass


def smax(x: Tensor, y: float) -> Tensor:
    """Take the element-wise maximum of a tensor and a scalar.
    :param x: Tensor
    :param y: float
    :returns: max(x, y)
    """
    pass


def argmin(x: Tensor, axis: Optional[int] = None):
    """Find and return the indices of the minimum values of a tensor along an axis.
    :param x: Tensor.
    :param axis: the axis along which to search.
    :returns: argmin(x)
    """

    pass


def argmax(x: Tensor, axis: Optional[int] = None):
    """Find and return the indices of the maximum values of a tensor along an axis.
    :param x: Tensor.
    :param axis: the axis along which to search.
    :returns: argmax(x)
    """

    pass


def cumsum(x: Tensor, axis: int, reverse: bool = False) -> Tensor:
    """Compute the cumulative sum of tensor values along a given axis.
    :param x: Tensor.
    :param axis: the axis along which to sum.
    :param reverse: whether or not to compute the cumulative sum in reverse order.
    """

    pass


def flip(x: Tensor, axis: int) -> Tensor:
    """Reverse the values of a tensor along a given axis.
    :param x: Tensor.
    :param axis: the axis along which to reverse the values.
    """

    pass


def abs(x: Tensor) -> Tensor:
    """Element-wise absolute value of a tensor.
    :param x: Tensor
    :returns: |x|
    """
    pass


def exp(x: Tensor) -> Tensor:
    """Element-wise exponential of a tensor.
    :param x: Tensor
    :returns: exp(x)
    """
    pass


def log(x: Tensor) -> Tensor:
    """Element-wise logarithm of a tensor.
    :param x: Tensor
    :returns: log(x)
    """
    pass


def sqrt(x: Tensor) -> Tensor:
    """Element-wise square-root of a tensor.
    :param x: Tensor
    :returns: sqrt(x)
    """
    pass


def logsumexp(x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """Compute the log of the sum of exponentials of provided tensor along the given axis.
    :param x: Tensor
    :param axis: (optional) the axis along which to sum the exponentiated tensor.
        Default is to sum over all entries.
    :returns: log(x)
    """
    pass


def digitize(x: Tensor, boundaries: Tensor) -> Tensor:
    """Digitize or "bucketize" the values of x, returning the bucket index for each element.
    :param x: Tensor
    :param boundaries: Tensor. The boundaries of the buckets to use for digitizing x.
    :returns: a tensor where each element has been replaced by the index of the bucket into which it falls.
    """
    pass


def arange(
    start: Optional[Union[int, float]] = None,
    stop: Union[int, float] = None,
    step: Union[int, float] = 1,
) -> Tensor:
    """Return evenly spaced values within a given interval.
    :param start: (optional) the (inclusive) starting value for the interval.
    :param stop: the (exclusive) stopping value for the interval.
    :param step: the increment to use when generating the values.
    """
    pass


def expand_dims(x: Tensor, axis: Union[int, Tuple[int, ...]]) -> Tensor:
    """Insert a new axis into the tensor at 'axis' position.
    :param x: Tensor.
    :param axis: the position in new tensor when the axis is placed.
    :returns: Tensor.
    """
    pass


def transpose(x: Tensor, dim0: int, dim1: int) -> Tensor:
    """Swap the given dimensions of the tensor x to produce it's transpose.
    :returns: Tensor
    """
    pass


def unique(
    x: Tensor, axis: int = None, return_index: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Find the unique values in a tensor and return them.
    :param x: Tensor.
    :param axis: the axis to search over for unique values.
    :param return_index: whether or not to also return the first index at which each unique value is found.
    :returns: Tensor
    """
    pass


def stack(
    tensors: List[Tensor],
    axis: int = 0,
) -> Tensor:
    """Join a list of tensors along a new axis.
    :param axis: the axis along which to join the tensors.
    :returns: Tensor
    """
    pass


def allclose(
    x: Tensor,
    y: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Determine whether or not two tensors are element-wise equal within a tolerance.
    :param x: Tensor,
    :param y: Tensor,
    :param rtol: the relative tolerance to use.
    :param atol: the absolute tolerance to use.
    """
    pass


def size(x: Tensor) -> int:
    """Compute the total size of a tensor, i.e. the number of elements across all axes.
    :param x: Tensor.
    :returns: the number of elements in x.
    """
    pass


def sign(x: Tensor) -> Tensor:
    """Return the element-wise signs of the input tensor.
    :param x: Tensor.
    :returns: sign(x)
    """
    pass


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Return elements from x or y depending on the 'condition' tensor.
    :param condition: a tensor of truthy/boolean elements (non-zeros evaluate to true).
    :param x: the matrix to retrieve elements from when 'condition' is True.
    :param y: the matrix to retrieve elements from when 'condition' is False.
    """

    pass


def all(x: Tensor, axis: Optional[int] = None) -> Union[bool, Tensor]:
    """Test whether all tensor elements are truthy.
    :param x: Tensor.
    :param axis: axis over which to restrict the 'and' operation.
    :returns: bool.
    """

    pass


def any(x: Tensor) -> bool:
    """Test whether any element of the tensor is truthy.
    :param x: Tensor.
    :returns: bool.
    """

    pass


def eye(d: int) -> Tensor:
    """Return the identity operator as a 2d array of dimension d.
    :param x: Tensor.
    :returns: Tensor.
    """

    pass


def solve(A: Tensor, b: Tensor) -> Tensor:
    """Solve the linear system Ax = b for the input 'x'.
    :param A: square matrix defining the linear system.
    :param b: the targets of the linear system.
    :returns: Tensor. x, the solution to the linear system.
    """

    pass


def isnan(x: Tensor) -> Tensor:
    """Return an new tensor where each element is a boolean indicating if that element of 'x' is 'nan'.
    :param x: Tensor.
    :returns: boolean Tensor.
    """

    pass


def floor(x: Tensor) -> Tensor:
    """Return the floor of the input element-wise.
    :param x: Tensor.
    :returns: a new Tensor whose elements are those of 'x' rounded down to the nearest integer.
    """

    pass


def ceil(x: Tensor) -> Tensor:
    """Return the ceiling of the input element-wise.
    :param x: Tensor.
    :returns: a new Tensor whose elements are those of 'x' rounded up to the nearest integer.
    """

    pass


def ravel(x: Tensor) -> Tensor:
    """Return a contiguous flattened tensor. Equivalent to x.reshape(-1), but often faster.
    :param x: Tensor.
    :returns: flattened version of x.
    """

    pass


def copy(x: Tensor) -> Tensor:
    """Return a copy of the provided tensor.
    :param x: Tensor.
    :returns: x copied into a new memory location.
    """

    pass


def isin(
    elements: Tensor,
    test_elements: Tensor,
    assume_unique: bool = False,
    invert: bool = False,
) -> Tensor:
    """Test if each element of elements is in test_elements.
    :param elements: the tensor of elements to search.
    :param test_elements: the tensor of elements to lookup.
    :param assume_unique: whether or not both 'elements' and 'test_elements' contain
        unique values.
    :param invert: invert the truth-value of the return values, so that False is returned
        for each element of 'test_elements' in 'elements'.
    :returns: tensor of booleans the same length as 'test_elements'.
    """

    pass


def logical_not(x: Tensor) -> Tensor:
    """Compute the logical not of the elements of the input tensor.
    :param x: the input tensor.
    :returns: not(x).
    """

    pass


def sort(x: Tensor, axis: Optional[int] = None) -> Tensor:
    """Return a sorted copy of the given tensor.
        If 'axis' is None, the tensor is flattened before sorting.
    :param x: the tensor to sort.
    :param axis: the axis along which to sort the tensor.
    :returns: a sorted copy of the tensor.
    """

    pass


def squeeze(x: Tensor) -> Tensor:
    """Removes all dimensions of input tensor with size one.
    :param x: tensor
    :returns: squeeze(x)
    """

    pass


# =============== #

# default to numpy
set_backend(NUMPY)
