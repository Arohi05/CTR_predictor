"""Python client for the CTRRegistry smart contract.

Three ways to run it (no code change needed, just environment variables):

1. Nothing set        -> an in-memory test chain (eth-tester). Zero setup, ideal for
                         demos and tests. State is lost when the program stops, but
                         `sync_ledger()` simply re-anchors every ledger block.
2. CTR_RPC_URL=http://127.0.0.1:8545
                      -> a local node such as `npx hardhat node` or `ganache`.
                         The first account of the node is used for transactions.
3. CTR_RPC_URL=<testnet url> and CTR_PRIVATE_KEY=<key of a funded test account>
                      -> a public testnet (e.g. Sepolia). Never commit the key and
                         never use a key that holds real money.

Optionally set CTR_CONTRACT_ADDRESS to reuse an already deployed contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from web3 import EthereumTesterProvider, Web3

from ledger import Ledger

ARTIFACT_PATH = Path(__file__).resolve().parent / "CTRRegistry.json"
ADDRESS_FILE = Path(__file__).resolve().parent.parent / "ledger" / "contract_address.json"


def _b32(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)


def _hex(b: bytes) -> str:
    return b.hex()


class ChainClient:
    def __init__(self, rpc_url: str | None = None, contract_address: str | None = None, private_key: str | None = None):
        rpc_url = rpc_url or os.getenv("CTR_RPC_URL")
        contract_address = contract_address or os.getenv("CTR_CONTRACT_ADDRESS")
        private_key = private_key or os.getenv("CTR_PRIVATE_KEY")

        if rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.w3.is_connected():
                raise ConnectionError(f"Cannot reach blockchain node at {rpc_url}")
            self.network = rpc_url
            self.persistent = True
        else:
            self.w3 = Web3(EthereumTesterProvider())
            self.network = "in-memory test chain"
            self.persistent = False

        self._account = self.w3.eth.account.from_key(private_key) if private_key else None
        self.sender = self._account.address if self._account else self.w3.eth.accounts[0]

        with open(ARTIFACT_PATH) as f:
            artifact = json.load(f)
        self._abi, self._bytecode = artifact["abi"], artifact["bytecode"]

        address = contract_address or self._saved_address()
        if address and self.w3.eth.get_code(Web3.to_checksum_address(address)):
            self.contract = self.w3.eth.contract(address=Web3.to_checksum_address(address), abi=self._abi)
        else:
            self.contract = self._deploy()

    # ------------------------------------------------------------ plumbing
    def _saved_address(self) -> str | None:
        if self.persistent and ADDRESS_FILE.exists():
            return json.loads(ADDRESS_FILE.read_text()).get("address")
        return None

    def _deploy(self):
        factory = self.w3.eth.contract(abi=self._abi, bytecode=self._bytecode)
        receipt = self._send(factory.constructor())
        contract = self.w3.eth.contract(address=receipt.contractAddress, abi=self._abi)
        if self.persistent:
            ADDRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            ADDRESS_FILE.write_text(json.dumps({"address": contract.address, "network": self.network}))
        return contract

    def _send(self, fn):
        """Send a transaction (signed locally if a private key is configured)."""
        if self._account:
            tx = fn.build_transaction(
                {
                    "from": self.sender,
                    "nonce": self.w3.eth.get_transaction_count(self.sender),
                    "chainId": self.w3.eth.chain_id,
                }
            )
            signed = self._account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            tx_hash = fn.transact({"from": self.sender})
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    @property
    def address(self) -> str:
        return self.contract.address

    # -------------------------------------------------------------- batches
    def batch_count(self) -> int:
        return self.contract.functions.batchCount().call()

    def anchor_block(self, block) -> str:
        """Anchor one ledger block. Returns the transaction hash."""
        receipt = self._send(
            self.contract.functions.anchorBatch(_b32(block.block_hash), _b32(block.merkle_root), block.record_count)
        )
        return "0x" + _hex(receipt.transactionHash)

    def sync_ledger(self, ledger: Ledger) -> int:
        """Anchor every ledger block the chain does not know yet, in order."""
        blocks, done = ledger.blocks(), self.batch_count()
        for block in blocks[done:]:
            self.anchor_block(block)
        return max(0, len(blocks) - done)

    def get_batch(self, index: int) -> dict:
        block_hash, root, count, ts = self.contract.functions.getBatch(index).call()
        return {"block_hash": _hex(block_hash), "merkle_root": _hex(root), "record_count": count, "timestamp": ts}

    def verify_ledger(self, ledger: Ledger) -> list[dict]:
        """Compare each local block with what the chain stored."""
        results, on_chain = [], self.batch_count()
        for b in ledger.blocks():
            if b.index >= on_chain:
                results.append({"index": b.index, "status": "not anchored yet"})
                continue
            stored = self.get_batch(b.index)
            ok = stored["block_hash"] == b.block_hash and stored["merkle_root"] == b.merkle_root
            results.append({"index": b.index, "status": "matches chain" if ok else "MISMATCH with chain"})
        return results

    # --------------------------------------------------------------- models
    def register_model(self, provenance: dict) -> str:
        receipt = self._send(
            self.contract.functions.registerModel(
                _b32(provenance["model_hash"]),
                _b32(provenance["dataset_hash"]),
                _b32(provenance["metrics_hash"]),
                provenance["version"],
            )
        )
        return "0x" + _hex(receipt.transactionHash)

    def find_model(self, model_hash: str) -> dict | None:
        found, index = self.contract.functions.findModel(_b32(model_hash)).call()
        if not found:
            return None
        mh, dh, xh, version, ts = self.contract.functions.getModel(index).call()
        return {
            "index": index,
            "model_hash": _hex(mh),
            "dataset_hash": _hex(dh),
            "metrics_hash": _hex(xh),
            "version": version,
            "timestamp": ts,
        }

