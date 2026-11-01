"""Pick the ledger from the LEDGER env var:  LEDGER=fake (default) | fabric"""
from __future__ import annotations

import os

from ledger.base import LedgerClient


def get_ledger(kind: str | None = None) -> LedgerClient:
    kind = (kind or os.getenv("LEDGER", "fake")).lower()
    if kind == "fabric":
        from ledger.fabric_client import FabricLedger
        return FabricLedger()
    if kind == "fake":
        from ledger.fake_ledger import FakeLedger
        return FakeLedger(os.getenv("FAKE_LEDGER_PATH", "data/fake_ledger.json"))
    raise ValueError(f"unknown LEDGER={kind!r} (use 'fake' or 'fabric')")