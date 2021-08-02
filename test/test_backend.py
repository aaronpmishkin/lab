"""
Test linear algebra backends against reference implementation (numpy).
"""

import unittest

import numpy as np
from scipy.special import logsumexp
import torch

from convex_nn import backend as be


class TestBackends(unittest.TestCase):
    """
    Test linear algebra backends for interoperability and correctness by
    comparison to a reference implementation (numpy).
    """

    def setUp(self):
        self.rng = np.random.default_rng(seed=778)

    def test_safe_divide(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            x = be.tensor([[1.0, 2, 0, 4, 5], [6, 7, 8, 9, 0]])
            y = be.to_np(x)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.safe_divide(x, x)),
                    np.divide(y, y, out=np.zeros_like(y), where=y != 0),
                ),
                f"{driver}: Safe divide did not match reference implementation for x ./ x.",
            )

            # more complex use case
            np_X = self.rng.standard_normal((2, 10, 10))
            np_X[:, :, 1] = np_X[:, :, 5] = 0
            np_col_norms = np.sum(np_X ** 2, axis=1, keepdims=True)

            X = be.tensor(np_X)
            col_norms = be.tensor(np_col_norms)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.safe_divide(X, col_norms)),
                    np.divide(
                        np_X,
                        np_col_norms,
                        out=np.zeros_like(np_X),
                        where=np_col_norms != 0,
                    ),
                ),
                f"{driver}: Safe divide did not match reference implementation for normalizing tensor columns.",
            )

    def test_concatenate(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            tensor_list = [
                be.tensor(self.rng.standard_normal((2, 4))) for i in range(5)
            ]
            np_tensor_list = [be.to_np(tensor) for tensor in tensor_list]

            # simple case without axis specified

            self.assertTrue(
                np.allclose(
                    be.concatenate(tensor_list), np.concatenate(np_tensor_list)
                ),
                f"{driver}: Concatenating columns (ie. stacking rows) of two matrices failed.",
            )

            # simple case with axis specified

            self.assertTrue(
                np.allclose(
                    be.concatenate(tensor_list, axis=1),
                    np.concatenate(np_tensor_list, axis=1),
                ),
                f"{driver}: Concatenating rows (ie. stacking rows) of two matrices failed.",
            )

            # concatenating tensors with different shapes
            tensor_list = [
                be.tensor(self.rng.standard_normal((2, i, 4))) for i in range(5)
            ]
            np_tensor_list = [be.to_np(tensor) for tensor in tensor_list]

            # concatenating along axis 1 should succeed:
            self.assertTrue(
                np.allclose(
                    be.concatenate(tensor_list, axis=1),
                    np.concatenate(np_tensor_list, axis=1),
                ),
                f"{driver}: Concatenating tensors with different shapes failed.",
            )

    def test_sum(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            np_X = self.rng.standard_normal((2, 10, 10))
            X = be.tensor(np_X)

            # sum entire tensor
            self.assertTrue(
                np.allclose(
                    be.sum(X),
                    np.sum(np_X),
                ),
                f"{driver}: Summing all elements of tensor did match reference implementation",
            )

            # sum entire tensor
            for axis in range(3):
                self.assertTrue(
                    np.allclose(
                        be.sum(X, axis=axis),
                        np.sum(np_X, axis=axis),
                    ),
                    f"{driver}: Summing axes of tensor did match reference implementation",
                )

    def test_numerical_ops(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            np_X = self.rng.standard_normal((2, 10, 10))
            np_y = self.rng.standard_normal((10))
            X = be.tensor(np_X)
            y = be.tensor(np_y)

            # element-wise multiplication
            self.assertTrue(
                np.allclose(
                    be.multiply(X, y),
                    np.multiply(np_X, np_y),
                ),
                f"{driver}: Element-wise multiplication did match reference implementation",
            )

            # element-wise division
            self.assertTrue(
                np.allclose(
                    be.divide(X, y),
                    np.divide(np_X, np_y),
                ),
                "Element-wise division did match reference implementation",
            )

            # matrix multiplication w/ broadcasting
            self.assertTrue(
                np.allclose(
                    be.matmul(X, y),
                    np.matmul(np_X, np_y),
                ),
                "Matmul with broadcasting did match reference implementation",
            )

    def test_creation_ops(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            shape = (3, 8, 2)
            np_x = np.zeros(shape)
            x = be.zeros(shape)

            self.assertTrue(
                np.allclose(be.to_np(x), np_x),
                f"{driver}: Creation of zeros matrix did not match reference.",
            )

            self.assertTrue(
                np.allclose(be.to_np(be.zeros_like(x)), np.zeros_like(np_x)),
                f"{driver}: Creation of zeros matrix did not match reference.",
            )

            self.assertTrue(
                np.allclose(be.to_np(be.ones_like(x)), np.ones_like(np_x)),
                f"{driver}: Creation of ones matrix did not match reference.",
            )

            np_y = np.ones(shape)
            y = be.ones(shape)

            self.assertTrue(
                np.allclose(be.to_np(y), np_y),
                f"{driver}: Creation of ones matrix did not match reference.",
            )

    def test_tensor_creation(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # creation from lists
            X = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
            list_tensor = be.tensor(X)

            # creation from numpy arrays
            np_tensor = be.tensor(np.array(X))

            self.assertTrue(
                np.allclose(be.to_np(np_tensor), be.to_np(list_tensor)),
                "Tensor created from list of lists did not match tensor created from numpy array.",
            )

    def test_extremes(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # arrays with the same shape
            np_X = self.rng.standard_normal((2, 10, 100))
            np_Y = self.rng.standard_normal((2, 10, 100))
            X = be.tensor(np_X)
            Y = be.tensor(np_Y)

            self.assertTrue(
                np.allclose(be.to_np(be.maximum(X, Y)), np.maximum(np_X, np_Y)),
                f"{driver}: Element-wise maximum of tensors did not match reference implementation.",
            )

            self.assertTrue(
                np.allclose(be.to_np(be.minimum(X, Y)), np.minimum(np_X, np_Y)),
                f"{driver}: Element-wise minimum of tensors did not match reference implementation.",
            )

            self.assertTrue(
                np.allclose(be.to_np(be.smax(X, 0.0)), np.maximum(np_X, 0)),
                f"{driver}: Maximum of tensor and scalar did not match reference implementation.",
            )

            self.assertTrue(
                np.allclose(be.to_np(be.smin(X, 0)), np.minimum(np_X, 0)),
                f"{driver}: Minimum of tensor and scalar did not match reference implementation.",
            )

    def test_diag(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = self.rng.standard_normal((100, 100))
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.diag(X)),
                    np.diag(np_X),
                ),
                f"{driver}: 'diag' did not extract the diagonal of the matrix.",
            )

            # check forming matrices from vectors
            np_x = self.rng.standard_normal(10)
            x = be.tensor(np_x)
            self.assertTrue(
                np.allclose(
                    be.to_np(be.diag(x)),
                    np.diag(np_x),
                ),
                f"{driver}: 'diag' did not create a diagonal matrix from a vector.",
            )

    def test_abs(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = self.rng.standard_normal((100, 100))
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.abs(X)),
                    np.abs(np_X),
                ),
                f"{driver}: 'abs' did not match the reference implementation.",
            )

    def test_sqrt(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = np.abs(self.rng.standard_normal((100, 100)))
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.sqrt(X)),
                    np.sqrt(np_X),
                ),
                f"{driver}: 'sqrt' did not match the reference implementation.",
            )

    def test_log(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = np.abs(self.rng.standard_normal((100, 100)))
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.log(X)),
                    np.log(np_X),
                ),
                f"{driver}: 'log' did not match the reference implementation.",
            )

    def test_exp(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = self.rng.standard_normal((100, 100))
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.exp(X)),
                    np.exp(np_X),
                ),
                f"{driver}: 'exp' did not match the reference implementation.",
            )

    def test_logsumexp(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = self.rng.standard_normal((100, 100))
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.logsumexp(X)),
                    logsumexp(np_X),
                ),
                f"{driver}: 'logsumexp' did not match the reference implementation when used without axis.",
            )

            self.assertTrue(
                np.allclose(
                    be.to_np(be.logsumexp(X, axis=1)),
                    logsumexp(np_X, axis=1),
                ),
                f"{driver}: 'logsumexp' did not match the reference implementation when specifying axis.",
            )

    def test_digitize(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = self.rng.standard_normal((100))
            X = be.tensor(np_X)

            np_boundaries = np.arange(-10, 10)

            self.assertTrue(
                np.all(
                    np.digitize(np_X, np_boundaries)
                    == be.to_np(be.digitize(X, be.tensor(np_boundaries)))
                )
            )

    def test_arange(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)
            # try simple range without increment or start.
            stop = 10

            self.assertTrue(
                np.allclose(
                    be.to_np(be.arange(stop)),
                    np.arange(stop),
                ),
                f"{driver}: 'arange' did not match the reference implementation when used with only a stopping point.",
            )
            start, stop = 10, 100

            self.assertTrue(
                np.allclose(
                    be.to_np(be.arange(start, stop)),
                    np.arange(start, stop),
                ),
                f"{driver}: 'arange' did not match the reference implementation when used with a starting and a stopping point.",
            )

            start, stop, step = 10, 100, 10

            self.assertTrue(
                np.allclose(
                    be.to_np(be.arange(start, stop, step)),
                    np.arange(start, stop, step),
                ),
                f"{driver}: 'arange' did not match the reference implementation when used with start, stop and step arguments.",
            )

    def test_expand_dims(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = self.rng.standard_normal((2, 10, 100))
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.expand_dims(X, axis=-1)),
                    np.expand_dims(np_X, axis=-1),
                ),
                "f{driver}: Expanding the final dimension with negative indexing failed to match reference implementation.",
            )

            self.assertTrue(
                np.allclose(
                    be.to_np(be.expand_dims(X, axis=0)),
                    np.expand_dims(np_X, axis=0),
                ),
                f"{driver}: Expanding the first (0) dimension failed to match reference implementation",
            )

            self.assertTrue(
                np.allclose(
                    be.to_np(be.expand_dims(X, axis=1)),
                    np.expand_dims(np_X, axis=1),
                ),
                f"{driver}: Expanding an internal dimension failed to match reference implementation.",
            )

    def test_transpose(self):
        # in this case we use the pytorch-style transpose, so the reference implementation is pytorch.
        for driver in be.BACKENDS:
            be.set_backend(driver)

            # check extracting matrix diagonals.
            np_X = self.rng.standard_normal((2, 10, 100))
            torch_X = torch.tensor(np_X)
            X = be.tensor(np_X)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.transpose(X, 0, 1)),
                    torch.transpose(torch_X, 0, 1),
                ),
                f"{driver}: Transposing the first two dimensions did not match reference implementation.",
            )

            self.assertTrue(
                np.allclose(
                    be.to_np(be.transpose(X, 0, -1)),
                    torch.transpose(torch_X, 0, -1),
                ),
                f"{driver}: Transposing the first and last dimensions did not match reference implementation.",
            )

    def test_stack(self):
        for driver in be.BACKENDS:
            be.set_backend(driver)

            np_tensors = [self.rng.standard_normal((2, 3, 5)) for i in range(5)]
            tensors = be.all_to_tensor(np_tensors)

            self.assertTrue(
                np.allclose(
                    be.to_np(be.stack(tensors, 0)),
                    np.stack(np_tensors, 0),
                ),
                f"{driver}: Transposing the first and last dimensions did not match reference implementation.",
            )


if __name__ == "__main__":
    unittest.main()
