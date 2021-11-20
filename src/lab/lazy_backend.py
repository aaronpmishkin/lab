"""
Backends which support lazy initialization.
This allows us to mimic the behavior of a module without the complexity of hot-swapping fields.
"""
from typing import Dict, Union, List, Tuple, Optional, Any, Iterable, overload

from .backend import Backend


class LazyBackend:
    """Interface for linear algebra backends which supports lazy initialization."""

    def __init__(self, backend: Backend):
        self.b = backend

    def set_backend(self, backend: Backend):
        self.b = backend

    def get_backend(self):
        return self.b

    # route all calls to the underlying backend.
    def __getattr__(self, attr: str) -> Any:
        return self.b.__getattribute__(attr)
