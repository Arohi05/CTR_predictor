"""Hashing utilities, Merkle trees and a tamper-evident, hash-chained ledger.

How it works
------------
1. Every prediction is a *record* (a small JSON object). Its fingerprint is
   SHA-256 of its canonical JSON (sorted keys, no spaces), so the same record
   always gives the same hash.
2. Records are collected in a pending pool. When a batch is *sealed*, the
   record hashes become the leaves of a Merkle tree and only its root is kept
   as the batch summary.
3. Each sealed batch becomes a block that also contains the hash of the
   previous block (hash chaining). Changing any old record changes its leaf,
   the Merkle root, the block hash, and therefore every block after it.
4. The block hash can be anchored on a blockchain (see blockchain/), so even
   the person running the app cannot silently rewrite history.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

GENESIS_HASH = "0" * 64
LEDGER_DIR = Path(__file__).resolve().parent / "ledger"


# --------------------------------------------------------------------------
# Hash helpers
# --------------------------------------------------------------------------
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def hash_record(record: dict) -> str:
    return sha256_hex(canonical_json(record))


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _pair(left: str, right: str) -> str:
    return sha256_hex(bytes.fromhex(left) + bytes.fromhex(right))


# --------------------------------------------------------------------------
# Merkle tree
# --------------------------------------------------------------------------
def merkle_root(leaves: list[str]) -> str:
    """Root of a Merkle tree over already-hashed leaves (hex strings)."""
    if not leaves:
        return GENESIS_HASH
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])  # duplicate the last node on odd levels
        level = [_pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: list[str], index: int) -> list[tuple[str, str]]:
    """Sibling hashes needed to prove leaves[index] belongs to the root.

    Each step is (sibling_hash, side) where side is "L" or "R".
    """
    if not 0 <= index < len(leaves):
        raise IndexError("leaf index out of range")
    proof, level, i = [], list(leaves), index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sibling = i ^ 1
        proof.append((level[sibling], "L" if sibling < i else "R"))
        level = [_pair(level[j], level[j + 1]) for j in range(0, len(level), 2)]
        i //= 2
    return proof


def verify_proof(leaf: str, proof: list[tuple[str, str]], root: str) -> bool:
    node = leaf
    for sibling, side in proof:
        node = _pair(sibling, node) if side == "L" else _pair(node, sibling)
    return node == root


# --------------------------------------------------------------------------
# Hash-chained ledger
# --------------------------------------------------------------------------
@dataclass
class Block:
    index: int
    timestamp: float
    prev_hash: str
    merkle_root: str
    record_count: int
    block_hash: str

    @staticmethod
    def compute_hash(index, timestamp, prev_hash, merkle_root, record_count) -> str:
        payload = {
            "index": index,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
            "merkle_root": merkle_root,
            "record_count": record_count,
        }
        return sha256_hex(canonical_json(payload))


class Ledger:
    """File-backed ledger: ledger/chain.json, ledger/pending.json, ledger/batch_N.json."""

    def __init__(self, directory: Path = LEDGER_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chain_path = self.dir / "chain.json"
        self.pending_path = self.dir / "pending.json"

    # ---- storage ----
    def _read(self, path: Path, default):
        if not path.exists():
            return default
        with open(path) as f:
            return json.load(f)

    def _write(self, path: Path, data) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def blocks(self) -> list[Block]:
        return [Block(**b) for b in self._read(self.chain_path, [])]

    def pending(self) -> list[dict]:
        return self._read(self.pending_path, [])

    def batch_records(self, index: int) -> list[dict]:
        return self._read(self.dir / f"batch_{index}.json", [])

    # ---- writing ----
    def add_record(self, record: dict) -> str:
        """Add a record to the pending pool and return its hash."""
        pending = self.pending()
        pending.append(record)
        self._write(self.pending_path, pending)
        return hash_record(record)

    def seal_batch(self) -> Block | None:
        """Turn all pending records into a new block. Returns None if empty."""
        records = self.pending()
        if not records:
            return None
        blocks = self.blocks()
        prev_hash = blocks[-1].block_hash if blocks else GENESIS_HASH
        root = merkle_root([hash_record(r) for r in records])
        index, ts = len(blocks), round(time.time(), 3)
        block = Block(
            index=index,
            timestamp=ts,
            prev_hash=prev_hash,
            merkle_root=root,
            record_count=len(records),
            block_hash=Block.compute_hash(index, ts, prev_hash, root, len(records)),
        )
        self._write(self.dir / f"batch_{index}.json", records)
        self._write(self.chain_path, [b.__dict__ for b in blocks] + [block.__dict__])
        self._write(self.pending_path, [])
        return block

    # ---- checking ----
    def verify(self) -> dict:
        """Recompute every hash. Reports the first block where something differs."""
        prev = GENESIS_HASH
        for b in self.blocks():
            records = self.batch_records(b.index)
            root = merkle_root([hash_record(r) for r in records])
            expected = Block.compute_hash(b.index, b.timestamp, b.prev_hash, root, len(records))
            if b.prev_hash != prev:
                return {"ok": False, "block": b.index, "reason": "broken link to previous block"}
            if root != b.merkle_root or len(records) != b.record_count:
                return {"ok": False, "block": b.index, "reason": "records were changed (Merkle root mismatch)"}
            if expected != b.block_hash:
                return {"ok": False, "block": b.index, "reason": "block header was changed"}
            prev = b.block_hash
        return {"ok": True, "block": None, "reason": "all blocks valid"}

    def prove_record(self, batch_index: int, record_index: int) -> dict:
        """Merkle proof that one record belongs to a sealed batch."""
        records = self.batch_records(batch_index)
        leaves = [hash_record(r) for r in records]
        return {
            "leaf": leaves[record_index],
            "proof": merkle_proof(leaves, record_index),
            "root": self.blocks()[batch_index].merkle_root,
        }
