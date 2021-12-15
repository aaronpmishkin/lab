"""
Backends which support lazy initialization.
This allows us to mimic the behavior of a module without the complexity of hot-swapping fields.
"""
from typing import Optional, Any

from .backend import Backend


class LazyBackend:
    """Interface for linear algebra backends which supports lazy initialization."""

    _b: Optional[Backend] = None

    def __init__(self, backend: Optional[Backend] = None):
        self._b = backend

    def set_backend(self, backend: Backend):
        self._b = backend

    def get_backend(self):
        if self._b is None:
            raise RuntimeError(
                "Tried to get Backend from uninitialized instance of LazyBackend."
            )

        return self._b

    # route all calls to the underlying backend.
    def __getattr__(self, attr: str) -> Any:
        assert self._b is not None

        return self._b.__getattribute__(attr)
