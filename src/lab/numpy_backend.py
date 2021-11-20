"""
Numpy implementation of backend for linear algebra operations.
"""
from typing import Union, Tuple, Any, Dict, Optional, List, cast, overload

import numpy as np
from scipy.special import logsumexp  # type: ignore
from scipy.linalg import solve  # type: ignore
import opt_einsum as oe  # type: ignore

from .backend import Backend
from .types import Tensor, TensorList, TensorType, DeviceEnum, DtypeEnum


class NumpyBackend(Backend):

    """Wrapper for linear algebra operations implemented by Numpy.
    :param device: the device on which to linear algebra computations. Typically "cpu" or "cuda".
    :param dtype: the default data type to use when creating tensors. Typically "float32" or "float64".
    :param use_autodiff: whether or not to leave reverse mode autodiff active (if supported).
    :param name: a name for the linear algebra backend. Defaults to `None`, in which case the backend is anonymous.
    :param seed: an optional seed for the default numpy random number generator.
    """

    dtype_map: Dict[DtypeEnum, TensorType] = {
        "float32": np.float32,
        "float64": np.float64,
    }

    def __init__(
        self,
        device: DeviceEnum = "cpu",
        dtype: DtypeEnum = "float32",
        use_autodiff: bool = False,
        name: Optional[str] = None,
        seed: int = 650
    ):
        super().__init__(device, dtype, use_autodiff, name, seed)

    # ===== Setters ===== #

    def set_device(self, device_name: DeviceEnum):
        """Set the default device for linear algebra operations.
        :param device_name: a string identifying the device to use.
        """
        if device_name != "cpu":
            raise ValueError(
                f"NumpyBackend does not support device {device_name}. It only supports CPU devices."
            )

    def set_default_dtype(self, dtype: DtypeEnum):
        """Set the default device for linear algebra operations.
        :param dtype: a string identifying the dtype to use.
        """
        self.default_dtype = self.dtype_map[dtype]

    def toggle_autodiff(self, use_autodiff: bool):
        """Toggle auto-diff engine.
        :param use_autodiff: whether or not to enable the autodiff engine.
        """
        if use_autodiff:
            raise ValueError("NumpyBackend does not support automatic differentiation.")

    # ===== Linear Algebra Methods ===== #

    # creation ops

    def tensor(self, x: Any, dtype: TensorType = None) -> Tensor:
        """Create a new tensor.
        :param x: an array-like object with data for the new tensor.
        :param dtype: the data type to use when constructing the tensor.
        :returns: a new Tensor object with supplied data and type.
        """
        if dtype is None:
            dtype = self.default_dtype

        assert isinstance(dtype, np.dtype)
        return np.array(x, dtype=dtype)

    def ravel(self, x: Tensor) -> Tensor:
        """Return a contiguous flattened tensor. Equivalent to x.reshape(-1), but often faster.
        :param x: Tensor.
        :returns: flattened version of x.
        """
        assert isinstance(x, np.ndarray)
        return np.ravel(x)

    def copy(self, x: Tensor) -> Tensor:
        """Return a copy of the provided tensor.
        :param x: Tensor.
        :returns: x copied into a new memory location.
        """
        assert isinstance(x, np.ndarray)
        return np.copy(x)

    def size(self, x: Tensor) -> int:
        """Compute the total size of a tensor, i.e. the number of elements across all axes.
        :param x: Tensor.
        :returns: the number of elements in x.
        """
        assert isinstance(x, np.ndarray)
        return np.size(x)

    def to_scalar(self, x: Union[Tensor, float]) -> float:
        """Cast a 1-element tensor into a floating point number.
        :param x: a tensor or floating point number.
        :returns: scalar value that was stored in x or x if it is already a floating point number.
        """
        if isinstance(x, np.ndarray):
            assert x.size == 1
            return x[0]

        assert isinstance(x, (float, int))
        return x

    def to_np(self, x: Tensor) -> np.ndarray:
        """Cast a given tensor into a numpy array on the CPU.
        Note: use 'tensor' to create new methods when working with the NumpyBackend.
        :param x: Tensor.
        :returns: np.array(x) or x is x is already a ndarray
        """
        if isinstance(x, np.ndarray):
            return x

        return np.array(x)

    def concatenate(self, tensors: TensorList, axis: int = 0) -> Tensor:
        """Join list of tensors along an exiting axis.
        :param tensors: list of tensors to join.
        :param axis: the along which to join the tensors.
        :returns: tensors concatenated along the given axis.
        """
        tensors = cast(List[np.ndarray], tensors)
        return np.concatenate(tensors, axis)

    def stack(
        self,
        tensors: TensorList,
        axis: int = 0,
    ) -> Tensor:
        """Join a list of tensors along a new axis.
        :param axis: the axis along which to join the tensors.
        :returns: Tensor
        """
        tensors = cast(List[np.ndarray], tensors)
        return np.stack(tensors, axis)

    def zeros(
        self, shape: Union[List[int], Tuple[int, ...]], dtype: TensorType = None
    ) -> Tensor:
        """Return a tensor of given shape filled with zeros.
        :param shape: the shape of the resulting tensor.
        :param dtype: the data type to use for the tensor.
        :returns: tensor filled with zeros of the desired shape.
        """
        dtype = self.default_dtype if dtype is None else dtype
        assert isinstance(dtype, np.dtype)

        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, x: Tensor) -> Tensor:
        """Return a tensor of zeros with the same shape and type as the input tensor.
        :param x: Tensor
        :returns: a tensor with the same shape and type as x, filled with zeros.
        """
        assert isinstance(x, np.ndarray)
        return np.zeros_like(x)

    def ones(
        self, shape: Union[List[int], Tuple[int, ...]], dtype: TensorType = None
    ) -> Tensor:
        """Return a tensor of given shape filled with ones.
        :param shape: the shape of the resulting tensor.
        :param dtype: the data type to use for the tensor.
        :returns: tensor filled with ones of the desired shape.
        """
        dtype = default_dtype["value"] if dtype is None else dtype
        assert isinstance(dtype, np.dtype)

        return np.ones(shape, dtype=dtype)

    def ones_like(self, x: Tensor) -> Tensor:
        """Return a tensor of ones with the same shape and type as the input tensor.
        :param x: Tensor
        :returns: a tensor with the same shape and type as x, filled with ones.
        """
        assert isinstance(x, np.ndarray)
        return np.ones_like(x)

    def diag(self, x: Tensor) -> Tensor:
        """Extract the diagonal of a tensor or construct a diagonal tensor.
        :param x: Tensor. If x is 2-d, then the diagonal of 'x' is extracted.
            If 'x' is 1-d, then 'Diag(x)' is returned.
        :returns: Tensor
        """
        assert isinstance(x, np.ndarray)
        return np.diag(x)

    def eye(self, d: int) -> Tensor:
        """Return the identity operator as a 2d array of dimension d.
        :param x: Tensor.
        :returns: Tensor.
        """
        return np.eye(d)

    @overload
    def arange(self, end: Union[int, float]) -> Tensor:
        ...

    @overload
    def arange(self, start: Union[int, float], stop: Union[int, float]) -> Tensor:
        ...

    @overload
    def arange(
        self, start: Union[int, float], stop: Union[int, float], step: Union[int, float]
    ) -> Tensor:
        ...

    def arange(
        self,
        start: Union[int, float],
        stop: Optional[Union[int, float]] = None,
        step: Optional[Union[int, float]] = None,
    ) -> Tensor:
        """Return evenly spaced values within a given interval.
        :param start: (optional) the (inclusive) starting value for the interval.
        :param stop: the (exclusive) stopping value for the interval.
        :param step: (optional) the increment to use when generating the values.
        """
        return np.arange(start, stop, step)

    def expand_dims(self, x: Tensor, axis: int) -> Tensor:
        """Insert a new axis into the tensor at 'axis' position.
        :param x: Tensor.
        :param axis: the position in new tensor when the axis is placed.
        :returns: Tensor.
        """
        assert isinstance(x, np.ndarray)
        return np.expand_dims(x, axis)

    def squeeze(self, x: Tensor) -> Tensor:
        """Removes all dimensions of input tensor with size one.
        :param x: tensor
        :returns: squeeze(x)
        """
        assert isinstance(x, np.ndarray)
        return np.squeeze(x)

    # math ops

    def einsum(self, path: str, *args) -> Tensor:
        """Optimized einsum operators.

        Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing
        contraction order for einsum-like expressions. Journal of Open Source Software, 2018, 3(26), 753
        DOI: https://doi.org/10.21105/joss.00753
        """
        return oe.contract(path, *args, backend="numpy")

    def sign(self, x: Tensor) -> Tensor:
        """Return the element-wise signs of the input tensor.
        :param x: Tensor.
        :returns: sign(x)
        """
        assert isinstance(x, np.ndarray)
        return np.sign(x)

    def safe_divide(self, x: Tensor, y: Tensor) -> Tensor:
        """Divide two tensors *safely*, where division by 0 is replaced with 0.
        :param x: Tensor.
        :param y: Tensor.
        :returns: x ./ y
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))

    def divide(self, x: Tensor, y: Tensor) -> Tensor:
        """Element-wise divide two tensors with broadcastable shapes.
        Note: this is *not* zero safe. Use 'safe_divide' when 0/0 is possible.
        :param x: Tensor
        :param y: Tensor
        :returns: x ./ y
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.divide(x, y)

    def multiply(self, x: Tensor, y: Tensor) -> Tensor:
        """Element-wise multiply two tensors with broadcastable shapes.
        :param x: Tensor
        :param y: Tensor
        :returns: x .* y
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.multiply(x, y)

    def abs(self, x: Tensor) -> Tensor:
        """Element-wise absolute value of a tensor.
        :param x: Tensor
        :returns: |x|
        """
        assert isinstance(x, np.ndarray)
        return np.abs(x)

    def exp(self, x: Tensor) -> Tensor:
        """Element-wise exponential of a tensor.
        :param x: Tensor
        :returns: exp(x)
        """
        assert isinstance(x, np.ndarray)
        return np.exp(x)

    def log(self, x: Tensor) -> Tensor:
        """Element-wise logarithm of a tensor.
        :param x: Tensor
        :returns: log(x)
        """
        assert isinstance(x, np.ndarray)
        return np.log(x)

    def sqrt(self, x: Tensor) -> Tensor:
        """Element-wise square-root of a tensor.
        :param x: Tensor
        :returns: sqrt(x)
        """
        assert isinstance(x, np.ndarray)
        return np.sqrt(x)

    def logsumexp(
        self, x: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Tensor:
        """Compute the log of the sum of exponentials of provided tensor along the given axis.
        :param x: Tensor
        :param axis: (optional) the axis along which to sum the exponentiated tensor.
            Default is to sum over all entries.
        :returns: log(x)
        """
        assert isinstance(x, np.ndarray)
        return logsumexp(x, axis)

    def digitize(self, x: Tensor, bins: Tensor) -> Tensor:
        """Digitize or "bucketize" the values of x, returning the bucket index for each element.
        :param x: Tensor
        :param bins: Tensor. The boundaries of the buckets to use for digitizing x.
        :returns: a tensor where each element has been replaced by the index of the bucket into which it falls.
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(bins, np.ndarray)

        return np.digitize(x, bins)

    def maximum(self, x: Tensor, y: Tensor) -> Tensor:
        """Element-wise maximum of the two input tensors.
        :param x: Tensor
        :param y: Tensor
        :returns: max(x, y)
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.maximum(x, y)

    def minimum(self, x: Tensor, y: Tensor) -> Tensor:
        """Element-wise minimum of the two input tensors.
        :param x: Tensor
        :param y: Tensor
        :returns: min(x, y)
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.minimum(x, y)

    def smax(self, x: Tensor, y: float) -> Tensor:
        """Take the element-wise maximum of a tensor and a scalar.
        :param x: Tensor
        :param y: float
        :returns: max(x, y)
        """
        assert isinstance(x, np.ndarray)
        return np.maximum(x, y)

    def smin(self, x: Tensor, y: float) -> Tensor:
        """Take the element-wise minimum of a tensor and a scalar.
        :param x: Tensor
        :param y: float
        :returns: min(x, y)
        """
        assert isinstance(x, np.ndarray)
        return np.minimum(x, y)

    def isnan(self, x: Tensor) -> Tensor:
        """Return an new tensor where each element is a boolean indicating if that element of 'x' is 'nan'.
        :param x: Tensor.
        :returns: boolean Tensor.
        """
        assert isinstance(x, np.ndarray)
        return np.isnan(x)

    def floor(self, x: Tensor) -> Tensor:
        """Return the floor of the input element-wise.
        :param x: Tensor.
        :returns: a new Tensor whose elements are those of 'x' rounded down to the nearest integer.
        """
        assert isinstance(x, np.ndarray)
        return np.floor(x)

    def ceil(self, x: Tensor) -> Tensor:
        """Return the ceiling of the input element-wise.
        :param x: Tensor.
        :returns: a new Tensor whose elements are those of 'x' rounded up to the nearest integer.
        """
        assert isinstance(x, np.ndarray)
        return np.ceil(x)

    def cumsum(self, x: Tensor, axis: int, reverse: bool = False) -> Tensor:
        """Compute the cumulative sum of tensor values along a given axis.
        :param x: Tensor.
        :param axis: the axis along which to sum.
        :param reverse: whether or not to compute the cumulative sum in reverse order.
        """
        assert isinstance(x, np.ndarray)
        if reverse:
            x = np.flip(x, axis=axis)

        res = np.cumsum(x, axis)

        if reverse:
            res = np.flip(res, axis=axis)

        return res

    # matrix ops

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        """Matrix product of two tensors.
        :param x: the first matrix.
        :param y: the second matrix.
        :returns: x @ y
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        return np.matmul(x, y)

    def dot(self, x: Tensor, y: Tensor) -> Tensor:
        """Euclidean inner-product of two vectors.
        :param x: the first (d,) vector.
        :param y: the second (d,) vector.
        :returns: <x, y>
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        return np.dot(x, y)

    def transpose(self, x: Tensor, dim0: int, dim1: int) -> Tensor:
        """Swap the given dimensions of the tensor x to produce it's transpose.
        :param x: the input tensor.
        :param dim0: the first dimension of the two dimensions to exchange.
        :param dim1: the first dimension of the two dimensions to exchange.
        :returns: tensor with the position of dimensions dim0, dim1 exchanged.
        """
        assert isinstance(x, np.ndarray)
        dims = np.arange(len(x.shape))
        dims[[dim0, dim1]] = [dim1, dim0]

        return np.transpose(x, dims)

    def solve(self, A: Tensor, b: Tensor) -> Tensor:
        """Solve the linear system Ax = b for the input 'x'.
        :param A: square matrix defining the linear system.
        :param b: the targets of the linear system.
        :returns: Tensor. x, the solution to the linear system.
        """
        assert isinstance(A, np.ndarray)
        assert isinstance(b, np.ndarray)
        return solve(A, b)

    def flip(self, x: Tensor, axis: int) -> Tensor:
        """Reverse the values of a tensor along a given axis.
        :param x: Tensor.
        :param axis: the axis along which to reverse the values.
        """
        assert isinstance(x, np.ndarray)
        return np.flip(x, axis)

    # reduction ops

    def sum(
        self,
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
        assert isinstance(x, np.ndarray)
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(
        self,
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
        assert isinstance(x, np.ndarray)
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: Tensor) -> Tensor:
        """Element-wise maximum of the input tensor.
        :param x: Tensor
        :returns: max(x)
        """
        assert isinstance(x, np.ndarray)
        return np.max(x)

    def min(self, x: Tensor) -> Tensor:
        """Element-wise minimum of the input tensor.
        :param x: Tensor
        :returns: max(x)
        """
        assert isinstance(x, np.ndarray)
        return np.min(x)

    def argmax(self, x: Tensor, axis: Optional[int] = None):
        """Find and return the indices of the maximum values of a tensor along an axis.
        :param x: Tensor.
        :param axis: the axis along which to search.
        :returns: argmax(x)
        """
        assert isinstance(x, np.ndarray)
        return np.argmax(x, axis)

    def argmin(self, x: Tensor, axis: Optional[int] = None):
        """Find and return the indices of the minimum values of a tensor along an axis.
        :param x: Tensor.
        :param axis: the axis along which to search.
        :returns: argmin(x)
        """
        assert isinstance(x, np.ndarray)
        return np.argmin(x, axis)

    def unique(
        self, x: Tensor, axis: Optional[int] = None, return_index: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Find the unique values in a tensor and return them.
        :param x: Tensor.
        :param axis: the axis to search over for unique values.
        :param return_index: whether or not to also return the first index at which each unique value is found.
        :returns: Tensor
        """
        assert isinstance(x, np.ndarray)
        return np.unique(x, axis=axis, return_index=return_index)

    # comparison ops

    def allclose(
        self,
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
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.allclose(x, y, rtol=rtol, atol=atol)

    def where(self, condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """Return elements from x or y depending on the 'condition' tensor.
        :param condition: a tensor of truthy/boolean elements (non-zeros evaluate to true).
        :param x: the matrix to retrieve elements from when 'condition' is True.
        :param y: the matrix to retrieve elements from when 'condition' is False.
        """
        assert isinstance(condition, np.ndarray)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        return np.where(condition, x, y)

    def all(self, x: Tensor, axis: Optional[int] = None) -> Union[bool, Tensor]:
        """Test whether all tensor elements are truthy.
        :param x: Tensor.
        :param axis: (optional) axis over which to reduce.
        :returns: bool or tensor of bool.
        """
        assert isinstance(x, np.ndarray)
        return np.all(x, axis=axis)

    def any(self, x: Tensor, axis: Optional[int] = None) -> Union[bool, Tensor]:
        """Test whether any element of the tensor is truthy.
        :param x: Tensor.
        :param axis: (optional) axis over which to reduce.
        :returns: bool or tensor of bool.
        """
        assert isinstance(x, np.ndarray)
        return np.any(x, axis=axis)

    def isin(
        self,
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
        assert isinstance(elements, np.ndarray)
        assert isinstance(test_elements, np.ndarray)

        return np.isin(
            elements, test_elements, assume_unique=assume_unique, invert=invert
        )

    def sort(self, x: Tensor, axis: Optional[int] = None) -> Tensor:
        """Return a sorted copy of the given tensor.
            If 'axis' is None, the tensor is flattened before sorting.
        :param x: the tensor to sort.
        :param axis: the axis along which to sort the tensor.
        :returns: a sorted copy of the tensor.
        """
        assert isinstance(x, np.ndarray)
        return np.sort(x, axis=axis)

    # logical ops

    def logical_not(self, x: Tensor) -> Tensor:
        """Compute the logical negation of the elements of the input tensor.
        :param x: the input tensor.
        :returns: not(x).
        """
        assert isinstance(x, np.ndarray)
        return np.logical_not(x)

    def logical_or(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the logical disjunction of the elements of the input tensor.
        :param x: input tensor.
        :param y: input tensor.
        :returns: the element-wise disjunction x v y.
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.logical_or(x, y)

    def logical_xor(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the logical exclusive disjunction of the elements of the input tensor.
        :param x: input tensor.
        :param y: input tensor.
        :returns: the element-wise exclusive disjunction x (+) y.
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.logical_xor(x, y)

    def logical_and(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the logical conjunction of the elements of the input tensor.
        :param x: input tensor.
        :param y: input tensor.
        :returns: the element-wise conjunction x ^ y.
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        return np.logical_and(x, y)
