"""
SHA-256 fingerprints.  h_x = SHA256(x)

Rule: every artifact is converted to a *canonical* byte representation
before hashing, so the same content always gives the same hash
(same key order in JSON, same dtype / byte order for arrays).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no spaces, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_array(arr: np.ndarray) -> str:
    """Hash a NumPy array including its shape and dtype (little-endian, C-order)."""
    a = np.ascontiguousarray(arr)
    a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    header = canonical_json({"shape": list(a.shape), "dtype": a.dtype.str})
    return sha256_bytes(header + b"\x00" + a.tobytes())


def sha256_text(text: str) -> str:
    # normalise line endings so Windows/Linux copies hash identically
    return sha256_bytes(text.replace("\r\n", "\n").encode("utf-8"))
