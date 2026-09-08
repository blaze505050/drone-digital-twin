"""
drone_sdk.validation_framework.manifest
=======================================
Dataset Provenance Manifest Schema & Integrity Verifier.
Implements Backlog Item B10 & PRD DAT-01/02.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import DataStatus


@dataclass
class ChannelMetadata:
    """Metadata describing an individual telemetry channel."""
    channel_name: str
    units: str
    coordinate_frame: str            # NED, FRD, ENU, SCALAR
    sampling_rate_hz: float
    description: str = ""


@dataclass
class DataManifest:
    """
    Provenance manifest accompanying any benchmark, flight log, or wind-tunnel dataset.
    Ensures fail-closed integrity and transparent empirical status.
    """
    dataset_id: str
    title: str
    description: str
    data_status: DataStatus = DataStatus.SYNTHETIC
    license: str = "CC-BY-4.0"
    source_url: str = ""
    file_sha256: str = ""
    clock_domain: str = "DEVICE_MONOTONIC"
    total_samples: int = 0
    duration_sec: float = 0.0
    nominal_rate_hz: float = 100.0
    channels: List[ChannelMetadata] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empirical(self) -> bool:
        """True only when data originates from physical sensors / flight hardware."""
        return bool(self.data_status == DataStatus.REAL)

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Calculate SHA256 checksum for a file on disk."""
        p = Path(file_path)
        if not p.is_file():
            return ""
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def verify_file_integrity(self, file_path: Path | str) -> bool:
        """Check if target file matches manifest SHA256 checksum."""
        if not self.file_sha256:
            return True
        computed = self.compute_sha256(file_path)
        return bool(computed.lower() == self.file_sha256.lower())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "title": self.title,
            "description": self.description,
            "data_status": self.data_status.value,
            "is_empirical": self.is_empirical,
            "license": self.license,
            "source_url": self.source_url,
            "file_sha256": self.file_sha256,
            "clock_domain": self.clock_domain,
            "total_samples": self.total_samples,
            "duration_sec": self.duration_sec,
            "nominal_rate_hz": self.nominal_rate_hz,
            "channels": [asdict(ch) for ch in self.channels],
            "metadata": self.metadata,
        }

    def save_json(self, json_path: Path | str) -> None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, json_path: Path | str) -> DataManifest:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = DataStatus(data.get("data_status", "synthetic"))
        channels = [
            ChannelMetadata(**ch) for ch in data.get("channels", [])
        ]
        return cls(
            dataset_id=data["dataset_id"],
            title=data["title"],
            description=data.get("description", ""),
            data_status=status,
            license=data.get("license", "CC-BY-4.0"),
            source_url=data.get("source_url", ""),
            file_sha256=data.get("file_sha256", ""),
            clock_domain=data.get("clock_domain", "DEVICE_MONOTONIC"),
            total_samples=data.get("total_samples", 0),
            duration_sec=data.get("duration_sec", 0.0),
            nominal_rate_hz=data.get("nominal_rate_hz", 100.0),
            channels=channels,
            metadata=data.get("metadata", {}),
        )
