"""
Off-chain storage (spec §4.6): the medical files and artifacts live HERE, never on-chain.

Rule used everywhere in the project:
    sha256 recorded on-chain  ==  SHA-256 of the exact bytes of the stored file
so verification is simply: re-read the file, re-hash, compare.

Layout:  data/offchain/<CASE-ID>/<ARTIFACT-ID>.<ext>
Files are written read-only for the owner (0o400) to discourage accidental edits.
(Possible extension: encrypt files at rest — the hash would then be computed on the plaintext.)
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.hashing import canonical_json, sha256_bytes


def serialize(payload: Any, kind: str) -> tuple[bytes, str]:
    """Canonical byte representation of an artifact -> (bytes, file extension)."""
    if kind == "json":
        return canonical_json(payload), "json"
    if kind == "text":
        return str(payload).replace("\r\n", "\n").encode("utf-8"), "txt"
    if kind == "npy":
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(payload), allow_pickle=False)
        return buf.getvalue(), "npy"
    if kind == "npz":       # several arrays (+ JSON strings) in one file
        buf = io.BytesIO()
        np.savez(buf, **{k: (np.asarray(json.dumps(v, sort_keys=True)) if isinstance(v, dict) else v)
                         for k, v in payload.items()})
        return buf.getvalue(), "npz"
    raise ValueError(f"unknown kind {kind}")


class OffchainStore:
    def __init__(self, root: str | Path = "data/offchain"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, case_id: str, artifact_id: str, data: bytes, ext: str) -> tuple[str, str]:
        """Write the file, return (uri relative to root, sha256 of the bytes)."""
        rel = Path(case_id) / f"{artifact_id}.{ext}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        path.write_bytes(data)
        os.chmod(path, 0o400)
        return rel.as_posix(), sha256_bytes(data)

    def path(self, uri: str) -> Path:
        p = (self.root / uri).resolve()
        if self.root.resolve() not in p.parents:      # no "../" escapes
            raise ValueError(f"invalid uri {uri}")
        return p

    def exists(self, uri: str) -> bool:
        return self.path(uri).is_file()

    def read(self, uri: str) -> bytes:
        return self.path(uri).read_bytes()

    def load(self, uri: str) -> Any:
        """Convenience loader for the UI / agents."""
        p = self.path(uri)
        if p.suffix == ".json":
            return json.loads(p.read_bytes())
        if p.suffix == ".txt":
            return p.read_text(encoding="utf-8")
        if p.suffix == ".npy":
            return np.load(p, allow_pickle=False)
        if p.suffix == ".npz":
            return dict(np.load(p, allow_pickle=False))
        raise ValueError(p.suffix)