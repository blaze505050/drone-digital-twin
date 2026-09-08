"""
drone_sdk.identification
========================
System identification, regressor conditioning checks, candidate parameter gating, and model rollback.
"""
from __future__ import annotations

from .identifier import (
    BoundedParameterIdentifier,
    IdentificationDataset,
    ParameterSet,
)

__all__ = [
    "IdentificationDataset",
    "ParameterSet",
    "BoundedParameterIdentifier",
]
