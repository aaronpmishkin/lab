"""
Test linear algebra backends with lazy implementation.
"""
import unittest

from parameterized import parameterized_class  # type: ignore
import numpy as np
from scipy.special import logsumexp  # type: ignore
import torch

import lab

from test_backend import TEST_GRID  # type: ignore

from lab.types import (
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

# ===== Tests ===== #


@parameterized_class(TEST_GRID)
class TestLazyBackends(unittest.TestCase):
    """
    Test lazy linear algebra backends.
    """

    backend: BackendEnum
    dtype: DtypeEnum

    def setUp(self):
        # setup backend
        lab.reset()
        self.lab = lab.get_backend(name="test_lazy", lazy=True)

        # instantiate the lazy backend.
        _ = lab.get_backend(name="test_lazy", impl=self.backend, dtype=self.dtype)

        self.rng = np.random.default_rng(seed=778)

    # creation ops

    def test_tensor(self):
        # creation from lists
        X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        list_tensor = self.lab.tensor(X)

        # creation from numpy arrays
        np_tensor = self.lab.tensor(np.array(X))

        self.assertTrue(
            np.allclose(self.lab.to_np(np_tensor), self.lab.to_np(list_tensor)),
            "Tensor created from list of lists did not match tensor created from numpy array.",
        )

    def test_ravel(self):
        pass  # TODO

    def test_copy(self):
        pass  # TODO

    def test_size(self):
        pass  # TODO

    def test_to_scalar(self):
        pass  # TODO

    def test_to_np(self):
        pass  # TODO

    def test_concatenate(self):

        tensor_list = [
            self.lab.tensor(self.rng.standard_normal((2, 4))) for i in range(5)
        ]
        np_tensor_list = [self.lab.to_np(tensor) for tensor in tensor_list]

        # simple case without axis specified

        self.assertTrue(
            np.allclose(
                self.lab.concatenate(tensor_list), np.concatenate(np_tensor_list)
            ),
            "Concatenating columns (ie. stacking rows) of two matrices failed.",
        )

        # simple case with axis specified

        self.assertTrue(
            np.allclose(
                self.lab.concatenate(tensor_list, axis=1),
                np.concatenate(np_tensor_list, axis=1),
            ),
            "Concatenating rows (ie. stacking rows) of two matrices failed.",
        )

        # concatenating tensors with different shapes
        tensor_list = [
            self.lab.tensor(self.rng.standard_normal((2, i, 4))) for i in range(5)
        ]
        np_tensor_list = [self.lab.to_np(tensor) for tensor in tensor_list]

        # concatenating along axis 1 should succeed:
        self.assertTrue(
            np.allclose(
                self.lab.concatenate(tensor_list, axis=1),
                np.concatenate(np_tensor_list, axis=1),
            ),
            "Concatenating tensors with different shapes failed.",
        )

    def test_stack(self):

        np_tensors = [self.rng.standard_normal((2, 3, 5)) for i in range(5)]
        tensors = self.lab.all_to_tensor(np_tensors)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.stack(tensors, 0)),
                np.stack(np_tensors, 0),
            ),
            "Transposing the first and last dimensions did not match reference implementation.",
        )

    def test_zeros_ones(self):
        shape = (3, 8, 2)
        np_x = np.zeros(shape)
        x = self.lab.zeros(shape)

        self.assertTrue(
            np.allclose(self.lab.to_np(x), np_x),
            "Creation of zeros matrix did not match reference.",
        )

        self.assertTrue(
            np.allclose(self.lab.to_np(self.lab.zeros_like(x)), np.zeros_like(np_x)),
            "Creation of zeros matrix did not match reference.",
        )

        self.assertTrue(
            np.allclose(self.lab.to_np(self.lab.ones_like(x)), np.ones_like(np_x)),
            "Creation of ones matrix did not match reference.",
        )

        np_y = np.ones(shape)
        y = self.lab.ones(shape)

        self.assertTrue(
            np.allclose(self.lab.to_np(y), np_y),
            "Creation of ones matrix did not match reference.",
        )

    def test_diag(self):
        np_X = self.rng.standard_normal((10, 10))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.diag(X)),
                np.diag(np_X),
            ),
            "'diag' did not extract the diagonal of the matrix.",
        )

        # check forming matrices from vectors
        np_x = self.rng.standard_normal(10)
        x = self.lab.tensor(np_x)
        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.diag(x)),
                np.diag(np_x),
            ),
            "'diag' did not create a diagonal matrix from a vector.",
        )

    def test_eye(self):
        pass  # TODO

    def test_arange(self):
        # try simple range without increment or start.
        stop = 10

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.arange(stop)),
                np.arange(stop),
            ),
            "'arange' did not match the reference implementation when used with only a stopping point.",
        )
        start, stop = 10, 100

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.arange(start, stop)),
                np.arange(start, stop),
            ),
            "'arange' did not match the reference implementation when used with a starting and a stopping point.",
        )

        start, stop, step = 10, 100, 10

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.arange(start, stop, step)),
                np.arange(start, stop, step),
            ),
            "'arange' did not match the reference implementation when used with start, stop and step arguments.",
        )

    def test_expand_dims(self):
        np_X = self.rng.standard_normal((2, 10, 100))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.expand_dims(X, axis=-1)),
                np.expand_dims(np_X, axis=-1),
            ),
            "Expanding the final dimension with negative indexing failed to match reference implementation.",
        )

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.expand_dims(X, axis=0)),
                np.expand_dims(np_X, axis=0),
            ),
            "Expanding the first (0) dimension failed to match reference implementation",
        )

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.expand_dims(X, axis=1)),
                np.expand_dims(np_X, axis=1),
            ),
            "Expanding an internal dimension failed to match reference implementation.",
        )

    def test_squeeze(self):
        pass  # TODO

    # math ops

    def test_sign(self):
        pass  # TODO

    def test_safe_divide(self):
        x = self.lab.tensor([[1.0, 2, 0, 4, 5], [6, 7, 8, 9, 0]])
        y = self.lab.to_np(x)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.safe_divide(x, x)),
                np.divide(y, y, out=np.zeros_like(y), where=y != 0),
            ),
            "Safe divide did not match reference implementation for x ./ x.",
        )

        # more complex use case
        np_X = self.rng.standard_normal((2, 10, 10))
        np_X[:, :, 1] = np_X[:, :, 5] = 0
        np_col_norms = np.sum(np_X ** 2, axis=1, keepdims=True)

        X = self.lab.tensor(np_X)
        col_norms = self.lab.tensor(np_col_norms)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.safe_divide(X, col_norms)),
                np.divide(
                    np_X,
                    np_col_norms,
                    out=np.zeros_like(np_X),
                    where=np_col_norms != 0,
                ),
            ),
            "Safe divide did not match reference implementation for normalizing tensor columns.",
        )

    def test_numerical_ops(self):
        np_X = self.rng.standard_normal((2, 10, 10))
        np_y = self.rng.standard_normal((10))
        X = self.lab.tensor(np_X)
        y = self.lab.tensor(np_y)

        # element-wise multiplication
        self.assertTrue(
            np.allclose(
                self.lab.multiply(X, y),
                np.multiply(np_X, np_y),
            ),
            "Element-wise multiplication did match reference implementation",
        )

        # element-wise division
        self.assertTrue(
            np.allclose(
                self.lab.divide(X, y),
                np.divide(np_X, np_y),
            ),
            "Element-wise division did match reference implementation",
        )

    def test_abs(self):
        np_X = self.rng.standard_normal((10, 10))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.abs(X)),
                np.abs(np_X),
            ),
            "'abs' did not match the reference implementation.",
        )

    def test_exp(self):
        np_X = self.rng.standard_normal((10, 10))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.exp(X)),
                np.exp(np_X),
            ),
            "'exp' did not match the reference implementation.",
        )

    def test_log(self):
        np_X = np.abs(self.rng.standard_normal((10, 10)))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.log(X)),
                np.log(np_X),
            ),
            "'log' did not match the reference implementation.",
        )

    def test_sqrt(self):
        np_X = np.abs(self.rng.standard_normal((10, 10)))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.sqrt(X)),
                np.sqrt(np_X),
            ),
            "'sqrt' did not match the reference implementation.",
        )

    def test_logsumexp(self):
        np_X = self.rng.standard_normal((10, 10))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.logsumexp(X)),
                logsumexp(np_X),
            ),
            "'logsumexp' did not match the reference implementation when used without axis.",
        )

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.logsumexp(X, axis=1)),
                logsumexp(np_X, axis=1),
            ),
            "'logsumexp' did not match the reference implementation when specifying axis.",
        )

    def test_digitize(self):
        np_X = self.rng.standard_normal((100))
        X = self.lab.tensor(np_X)

        np_boundaries = np.arange(-10, 10)

        self.assertTrue(
            np.all(
                np.digitize(np_X, np_boundaries)
                == self.lab.to_np(self.lab.digitize(X, self.lab.tensor(np_boundaries)))
            )
        )

    def test_maximum_minimum(self):
        # arrays with the same shape
        np_X = self.rng.standard_normal((2, 10, 100))
        np_Y = self.rng.standard_normal((2, 10, 100))
        X = self.lab.tensor(np_X)
        Y = self.lab.tensor(np_Y)

        self.assertTrue(
            np.allclose(self.lab.to_np(self.lab.maximum(X, Y)), np.maximum(np_X, np_Y)),
            "Element-wise maximum of tensors did not match reference implementation.",
        )

        self.assertTrue(
            np.allclose(self.lab.to_np(self.lab.minimum(X, Y)), np.minimum(np_X, np_Y)),
            "Element-wise minimum of tensors did not match reference implementation.",
        )

    def test_smax_smin(self):
        # arrays with the same shape
        np_X = self.rng.standard_normal((2, 10, 100))
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(self.lab.to_np(self.lab.smax(X, 0.0)), np.maximum(np_X, 0)),
            "Maximum of tensor and scalar did not match reference implementation.",
        )

        self.assertTrue(
            np.allclose(self.lab.to_np(self.lab.smin(X, 0)), np.minimum(np_X, 0)),
            "Minimum of tensor and scalar did not match reference implementation.",
        )

    def test_isnan(self):
        pass  # TODO

    def test_floor(self):
        pass  # TODO

    def test_ceil(self):
        pass  # TODO

    def test_cumsum(self):
        pass  # TODO

    # matrix ops

    def test_matmul(self):
        np_X = self.rng.standard_normal((2, 10, 10))
        np_y = self.rng.standard_normal((10))
        X = self.lab.tensor(np_X)
        y = self.lab.tensor(np_y)

        # matrix multiplication w/ broadcasting
        self.assertTrue(
            np.allclose(
                self.lab.matmul(X, y),
                np.matmul(np_X, np_y),
            ),
            "Matmul with broadcasting did match reference implementation",
        )

    def test_dot(self):
        pass  # TODO

    def test_transpose(self):
        # in this case we use the pytorch-style transpose, so the reference implementation is pytorch.
        np_X = self.rng.standard_normal((2, 10, 100))
        torch_X = torch.tensor(np_X)
        X = self.lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.transpose(X, 0, 1)),
                torch.transpose(torch_X, 0, 1),
            ),
            "Transposing the first two dimensions did not match reference implementation.",
        )

        self.assertTrue(
            np.allclose(
                self.lab.to_np(self.lab.transpose(X, 0, -1)),
                torch.transpose(torch_X, 0, -1),
            ),
            "Transposing the first and last dimensions did not match reference implementation.",
        )

    def test_solve(self):
        pass  # TODO

    def test_flip(self):
        pass  # TODO

    # reduction ops

    def test_sum(self):
        np_X = self.rng.standard_normal((2, 10, 10))
        X = self.lab.tensor(np_X)

        # sum entire tensor
        self.assertTrue(
            np.allclose(
                self.lab.sum(X),
                np.sum(np_X),
            ),
            "Summing all elements of tensor did match reference implementation",
        )

        # sum entire tensor
        for axis in range(3):
            self.assertTrue(
                np.allclose(
                    self.lab.sum(X, axis=axis),
                    np.sum(np_X, axis=axis),
                ),
                "Summing axes of tensor did match reference implementation",
            )

    def test_mean(self):
        pass  # TODO

    def test_max_min(self):
        pass  # TODO

    def test_argmax_argmin(self):
        pass  # TODO

    def test_unique(self):
        pass  # TODO

    # comparison ops

    def test_allclose(self):
        pass  # TODO

    def test_where(self):
        pass  # TODO

    def test_all(self):
        pass  # TODO

    def test_any(self):
        pass  # TODO

    def test_isin(self):
        pass  # TODO

    def test_sort(self):
        pass  # TODO

    # logical ops

    def test_logical(self):
        pass  # TODO


if __name__ == "__main__":
    unittest.main()
