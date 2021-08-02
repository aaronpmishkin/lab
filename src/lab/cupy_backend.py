"""
CuPy linear algebra backend. Defaults to NumPy if CuPy is not available.
"""

import numpy as np

# try to import CuPy and default to NumPy if package is not installed.
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = np

    def asnumpy(x: np.ndarray) -> np.ndarray:
        return x

    cp.asnumpy = asnumpy


def cp_safe_divide(x: cp.ndarray, y: cp.ndarray):
    return cp.divide(x, y, out=cp.zeros_like(x), where=(y != 0))


def cp_transpose(x: cp.ndarray, dim0: int, dim1: int) -> cp.ndarray:
    dims = cp.arange(len(x.shape))
    dims[[dim0, dim1]] = [dim1, dim0]

    return cp.transpose(x, dims)


cp_map = {
    "safe_divide": cp_safe_divide,
    "concatenate": cp.concatenate,
    "sum": cp.sum,
    "multiply": cp.multiply,
    "divide": cp.divide,
    "matmul": cp.matmul,
    "zeros": cp.zeros,
    "zeros_like": cp.zeros_like,
    "ones": cp.ones,
    "ones_like": cp.ones_like,
    "tensor": cp.array,
    "maximum": cp.maximum,
    "minimum": cp.minimum,
    "smax": cp.maximum,
    "smin": cp.minimum,
    "diag": cp.diag,
    "abs": cp.abs,
    "sqrt": cp.sqrt,
    "arange": cp.arange,
    "expand_dims": cp.expand_dims,
    "transpose": cp_transpose,
}
