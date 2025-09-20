"""Application package for the ID card extraction API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .main import app as fastapi_app


def __getattr__(name: str):
    if name == "app":
        from .main import app as fastapi_app  # local import to avoid importing FastAPI eagerly

        return fastapi_app
    raise AttributeError(f"module 'api' has no attribute {name!r}")


__all__ = ["app"]
