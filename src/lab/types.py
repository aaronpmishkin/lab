"""
Types and global constants for LAB.
"""
from typing import Literal, List, Union

import numpy as np
import torch

# ===== constants ===== #

# backend

TORCH = "torch"
NUMPY = "numpy"
CUPY = "cupy"

BACKENDS = [TORCH, NUMPY]

# devices

CPU = "cpu"
CUDA = "cuda"

# dtypes

FLOAT32: str = "float32"
FLOAT64: str = "float64"
INT32: str = "INT32"
INT64: str = "INT64"

DTYPES: List[str] = [FLOAT32, FLOAT64, INT32, INT64]

# ===== types ===== #

BackendEnum = Literal["torch", "numpy", "cupy"]
DeviceEnum = Literal["cpu", "cuda"]
DtypeEnum = Literal["float32", "float64"]

# Tensors
Tensor = Union[np.ndarray, torch.Tensor]
TensorList = Union[List[np.ndarray], List[torch.Tensor]]
TensorType = Union[torch.dtype, np.dtype, type]
