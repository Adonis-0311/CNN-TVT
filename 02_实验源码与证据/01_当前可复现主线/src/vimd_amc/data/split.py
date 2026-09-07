"""Source-level split identities and manifest checks."""

from __future__ import annotations

from hashlib import blake2b, sha256
import json
from typing import Iterable


_SPLIT_OFFSETS = {
    "train": 10_000_000,
    "validation": 20_000_000,
    "test": 30_000_000,
    "unseen_jammer": 40_000_000,
    "unseen_speed": 50_000_000,
    "hard": 60_000_000,
    "clean": 70_000_000,
    "grid": 80_000_000,
    "trajectory": 90_000_000,
    "unseen_channel": 100_000_000,
    "unseen_speed_and_channel": 110_000_000,
    "unseen_mobility": 120_000_000,
}


def stable_seed(*parts: object, bits: int = 63) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(blake2b(payload, digest_size=8).digest(), "little")
    return value & ((1 << bits) - 1)


def source_sequence_id(split: str, index: int, master_seed: int) -> int:
    if split not in _SPLIT_OFFSETS:
        raise ValueError(f"Unknown split: {split}")
    if index < 0:
        raise ValueError("index must be non-negative")
    # The large split prefix makes accidental overlap visible in plain manifests.
    return _SPLIT_OFFSETS[split] + master_seed * 1_000_000 + index


def assert_disjoint_source_ids(*groups: Iterable[int]) -> None:
    materialized = [set(group) for group in groups]
    for left in range(len(materialized)):
        for right in range(left + 1, len(materialized)):
            overlap = materialized[left].intersection(materialized[right])
            if overlap:
                example = sorted(overlap)[:5]
                raise AssertionError(f"source-sequence leakage detected: {example}")


def manifest_digest(records: list[dict[str, object]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()
