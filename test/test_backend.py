"""
Test linear algebra backends against reference implementation (numpy).
"""

import unittest

from parameterized import parameterized_class  # type: ignore
import numpy as np
from scipy.special import logsumexp  # type: ignore
import torch

import lab


@parameterized_class(lab.TEST_GRID)
class TestBackends(unittest.TestCase):
    """
    Test linear algebra backends for interoperability and correctness by
    comparison to a reference implementation (numpy).
    """

    def setUp(self):
        # setup backend
        lab.set_backend(self.backend)
        lab.set_dtype(self.dtype)

        self.rng = np.random.default_rng(seed=778)

    def test_safe_divide(self):
        x = lab.tensor([[1.0, 2, 0, 4, 5], [6, 7, 8, 9, 0]])
        y = lab.to_np(x)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.safe_divide(x, x)),
                np.divide(y, y, out=np.zeros_like(y), where=y != 0),
            ),
            "Safe divide did not match reference implementation for x ./ x.",
        )

        # more complex use case
        np_X = self.rng.standard_normal((2, 10, 10))
        np_X[:, :, 1] = np_X[:, :, 5] = 0
        np_col_norms = np.sum(np_X ** 2, axis=1, keepdims=True)

        X = lab.tensor(np_X)
        col_norms = lab.tensor(np_col_norms)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.safe_divide(X, col_norms)),
                np.divide(
                    np_X,
                    np_col_norms,
                    out=np.zeros_like(np_X),
                    where=np_col_norms != 0,
                ),
            ),
            "Safe divide did not match reference implementation for normalizing tensor columns.",
        )

    def test_concatenate(self):

        tensor_list = [lab.tensor(self.rng.standard_normal((2, 4))) for i in range(5)]
        np_tensor_list = [lab.to_np(tensor) for tensor in tensor_list]

        # simple case without axis specified

        self.assertTrue(
            np.allclose(lab.concatenate(tensor_list), np.concatenate(np_tensor_list)),
            "Concatenating columns (ie. stacking rows) of two matrices failed.",
        )

        # simple case with axis specified

        self.assertTrue(
            np.allclose(
                lab.concatenate(tensor_list, axis=1),
                np.concatenate(np_tensor_list, axis=1),
            ),
            "Concatenating rows (ie. stacking rows) of two matrices failed.",
        )

        # concatenating tensors with different shapes
        tensor_list = [
            lab.tensor(self.rng.standard_normal((2, i, 4))) for i in range(5)
        ]
        np_tensor_list = [lab.to_np(tensor) for tensor in tensor_list]

        # concatenating along axis 1 should succeed:
        self.assertTrue(
            np.allclose(
                lab.concatenate(tensor_list, axis=1),
                np.concatenate(np_tensor_list, axis=1),
            ),
            "Concatenating tensors with different shapes failed.",
        )

    def test_sum(self):
        np_X = self.rng.standard_normal((2, 10, 10))
        X = lab.tensor(np_X)

        # sum entire tensor
        self.assertTrue(
            np.allclose(
                lab.sum(X),
                np.sum(np_X),
            ),
            "Summing all elements of tensor did match reference implementation",
        )

        # sum entire tensor
        for axis in range(3):
            self.assertTrue(
                np.allclose(
                    lab.sum(X, axis=axis),
                    np.sum(np_X, axis=axis),
                ),
                "Summing axes of tensor did match reference implementation",
            )

    def test_numerical_ops(self):
        np_X = self.rng.standard_normal((2, 10, 10))
        np_y = self.rng.standard_normal((10))
        X = lab.tensor(np_X)
        y = lab.tensor(np_y)

        # element-wise multiplication
        self.assertTrue(
            np.allclose(
                lab.multiply(X, y),
                np.multiply(np_X, np_y),
            ),
            "Element-wise multiplication did match reference implementation",
        )

        # element-wise division
        self.assertTrue(
            np.allclose(
                lab.divide(X, y),
                np.divide(np_X, np_y),
            ),
            "Element-wise division did match reference implementation",
        )

        # matrix multiplication w/ broadcasting
        self.assertTrue(
            np.allclose(
                lab.matmul(X, y),
                np.matmul(np_X, np_y),
            ),
            "Matmul with broadcasting did match reference implementation",
        )

    def test_creation_ops(self):
        shape = (3, 8, 2)
        np_x = np.zeros(shape)
        x = lab.zeros(shape)

        self.assertTrue(
            np.allclose(lab.to_np(x), np_x),
            "Creation of zeros matrix did not match reference.",
        )

        self.assertTrue(
            np.allclose(lab.to_np(lab.zeros_like(x)), np.zeros_like(np_x)),
            "Creation of zeros matrix did not match reference.",
        )

        self.assertTrue(
            np.allclose(lab.to_np(lab.ones_like(x)), np.ones_like(np_x)),
            "Creation of ones matrix did not match reference.",
        )

        np_y = np.ones(shape)
        y = lab.ones(shape)

        self.assertTrue(
            np.allclose(lab.to_np(y), np_y),
            "Creation of ones matrix did not match reference.",
        )

    def test_tensor_creation(self):
        # creation from lists
        X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        list_tensor = lab.tensor(X)

        # creation from numpy arrays
        np_tensor = lab.tensor(np.array(X))

        self.assertTrue(
            np.allclose(lab.to_np(np_tensor), lab.to_np(list_tensor)),
            "Tensor created from list of lists did not match tensor created from numpy array.",
        )

    def test_extremes(self):
        # arrays with the same shape
        np_X = self.rng.standard_normal((2, 10, 100))
        np_Y = self.rng.standard_normal((2, 10, 100))
        X = lab.tensor(np_X)
        Y = lab.tensor(np_Y)

        self.assertTrue(
            np.allclose(lab.to_np(lab.maximum(X, Y)), np.maximum(np_X, np_Y)),
            "Element-wise maximum of tensors did not match reference implementation.",
        )

        self.assertTrue(
            np.allclose(lab.to_np(lab.minimum(X, Y)), np.minimum(np_X, np_Y)),
            "Element-wise minimum of tensors did not match reference implementation.",
        )

        self.assertTrue(
            np.allclose(lab.to_np(lab.smax(X, 0.0)), np.maximum(np_X, 0)),
            "Maximum of tensor and scalar did not match reference implementation.",
        )

        self.assertTrue(
            np.allclose(lab.to_np(lab.smin(X, 0)), np.minimum(np_X, 0)),
            "Minimum of tensor and scalar did not match reference implementation.",
        )

    def test_diag(self):
        np_X = self.rng.standard_normal((10, 10))
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.diag(X)),
                np.diag(np_X),
            ),
            "'diag' did not extract the diagonal of the matrix.",
        )

        # check forming matrices from vectors
        np_x = self.rng.standard_normal(10)
        x = lab.tensor(np_x)
        self.assertTrue(
            np.allclose(
                lab.to_np(lab.diag(x)),
                np.diag(np_x),
            ),
            "'diag' did not create a diagonal matrix from a vector.",
        )

    def test_abs(self):
        np_X = self.rng.standard_normal((10, 10))
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.abs(X)),
                np.abs(np_X),
            ),
            "'abs' did not match the reference implementation.",
        )

    def test_sqrt(self):
        np_X = np.abs(self.rng.standard_normal((10, 10)))
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.sqrt(X)),
                np.sqrt(np_X),
            ),
            "'sqrt' did not match the reference implementation.",
        )

    def test_log(self):
        np_X = np.abs(self.rng.standard_normal((10, 10)))
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.log(X)),
                np.log(np_X),
            ),
            "'log' did not match the reference implementation.",
        )

    def test_exp(self):
        np_X = self.rng.standard_normal((10, 10))
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.exp(X)),
                np.exp(np_X),
            ),
            "'exp' did not match the reference implementation.",
        )

    def test_logsumexp(self):
        np_X = self.rng.standard_normal((10, 10))
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.logsumexp(X)),
                logsumexp(np_X),
            ),
            "'logsumexp' did not match the reference implementation when used without axis.",
        )

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.logsumexp(X, axis=1)),
                logsumexp(np_X, axis=1),
            ),
            "'logsumexp' did not match the reference implementation when specifying axis.",
        )

    def test_digitize(self):
        np_X = self.rng.standard_normal((100))
        X = lab.tensor(np_X)

        np_boundaries = np.arange(-10, 10)

        self.assertTrue(
            np.all(
                np.digitize(np_X, np_boundaries)
                == lab.to_np(lab.digitize(X, lab.tensor(np_boundaries)))
            )
        )

    def test_arange(self):
        # try simple range without increment or start.
        stop = 10

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.arange(stop)),
                np.arange(stop),
            ),
            "'arange' did not match the reference implementation when used with only a stopping point.",
        )
        start, stop = 10, 100

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.arange(start, stop)),
                np.arange(start, stop),
            ),
            "'arange' did not match the reference implementation when used with a starting and a stopping point.",
        )

        start, stop, step = 10, 100, 10

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.arange(start, stop, step)),
                np.arange(start, stop, step),
            ),
            "'arange' did not match the reference implementation when used with start, stop and step arguments.",
        )

    def test_expand_dims(self):
        np_X = self.rng.standard_normal((2, 10, 100))
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.expand_dims(X, axis=-1)),
                np.expand_dims(np_X, axis=-1),
            ),
            "Expanding the final dimension with negative indexing failed to match reference implementation.",
        )

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.expand_dims(X, axis=0)),
                np.expand_dims(np_X, axis=0),
            ),
            "Expanding the first (0) dimension failed to match reference implementation",
        )

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.expand_dims(X, axis=1)),
                np.expand_dims(np_X, axis=1),
            ),
            "Expanding an internal dimension failed to match reference implementation.",
        )

    def test_transpose(self):
        # in this case we use the pytorch-style transpose, so the reference implementation is pytorch.
        np_X = self.rng.standard_normal((2, 10, 100))
        torch_X = torch.tensor(np_X)
        X = lab.tensor(np_X)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.transpose(X, 0, 1)),
                torch.transpose(torch_X, 0, 1),
            ),
            "Transposing the first two dimensions did not match reference implementation.",
        )

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.transpose(X, 0, -1)),
                torch.transpose(torch_X, 0, -1),
            ),
            "Transposing the first and last dimensions did not match reference implementation.",
        )

    def test_stack(self):

        np_tensors = [self.rng.standard_normal((2, 3, 5)) for i in range(5)]
        tensors = lab.all_to_tensor(np_tensors)

        self.assertTrue(
            np.allclose(
                lab.to_np(lab.stack(tensors, 0)),
                np.stack(np_tensors, 0),
            ),
            "Transposing the first and last dimensions did not match reference implementation.",
        )


if __name__ == "__main__":
    unittest.main()
