"""Python client for the CTRRegistry smart contract.

Three ways to run it (no code change needed, just environment variables):

1. Nothing set        -> an in-memory test chain (eth-tester). Zero setup, ideal for
                         demos and tests. State is lost when the program stops, but
                         `sync_ledger()` simply re-anchors every ledger block.
2. CTR_RPC_URL=http://127.0.0.1:8545
                      -> a local node such as `npx hardhat node` or `ganache`.
                         The first account of the node is used for transactions.
3. CTR_RPC_URL=<testnet url> and CTR_PRIVATE_KEY=<key of a funded test account>
                      -> a public testnet (e.g. Sepolia). Transactions are signed on your
                         computer; the key never leaves it. Never commit the key and never
                         use a key that holds real money. Ethereum mainnet is refused.

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

MAINNET_CHAIN_ID = 1
NETWORK_NAMES = {11155111: "Sepolia testnet", 31337: "local Hardhat/Anvil node", 1337: "local Ganache node"}
EXPLORERS = {11155111: "https://sepolia.etherscan.io"}


class ChainError(RuntimeError):
    """A blockchain connection or account problem, worded so the user knows what to do."""


def _b32(hex_str: str) -> bytes:
    return bytes.fromhex(hex_str)


def _hex(b: bytes) -> str:
    return b.hex()


class ChainClient:
    def __init__(
        self,
        rpc_url: str | None = None,
        contract_address: str | None = None,
        private_key: str | None = None,
        w3: Web3 | None = None,
    ):
        rpc_url = rpc_url or os.getenv("CTR_RPC_URL")
        contract_address = contract_address or os.getenv("CTR_CONTRACT_ADDRESS")
        private_key = (private_key or os.getenv("CTR_PRIVATE_KEY") or "").strip() or None

        if w3 is not None:  # a ready-made connection (used by the tests)
            self.w3, self.network, self.persistent = w3, "custom provider", False
        elif rpc_url:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
            if not self.w3.is_connected():
                raise ChainError(f"Cannot reach the blockchain node at {rpc_url}. Check the URL and your internet connection.")
            self.network, self.persistent = rpc_url, True
        else:
            self.w3 = Web3(EthereumTesterProvider())
            self.network, self.persistent = "in-memory test chain", False

        self.chain_id = self.w3.eth.chain_id
        if self.chain_id == MAINNET_CHAIN_ID:
            raise ChainError("This node is Ethereum mainnet, which uses real money. This project only supports test networks and refuses to continue.")
        if self.persistent:
            self.network = NETWORK_NAMES.get(self.chain_id, f"chain {self.chain_id}")

        try:
            self._account = self.w3.eth.account.from_key(private_key) if private_key else None
        except ValueError as e:  # never put the key itself in a message
            raise ChainError("CTR_PRIVATE_KEY is not a valid private key (expected 64 hexadecimal characters).") from e
        self.sender = self._account.address if self._account else self.w3.eth.accounts[0]
        if self.persistent and self.balance_wei() == 0:
            raise ChainError(f"Account {self.sender} has no test ETH on {self.network}. Get some from a faucet, wait for it to arrive, then try again.")

        with open(ARTIFACT_PATH) as f:
            artifact = json.load(f)
        self._abi, self._bytecode = artifact["abi"], artifact["bytecode"]

        address = contract_address or self._saved_address()
        self.contract = None
        if address:
            try:
                checksum = Web3.to_checksum_address(address)
            except ValueError as e:
                raise ChainError("The contract address is not valid.") from e
            if self.w3.eth.get_code(checksum):
                self.contract = self.w3.eth.contract(address=checksum, abi=self._abi)
        self.reused_contract = self.contract is not None
        if self.contract is None:
            self.contract = self._deploy()

    # ------------------------------------------------------------ plumbing
    def _read_saved(self) -> dict:
        try:
            return json.loads(ADDRESS_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def _saved_address(self) -> str | None:
        return self._read_saved().get(str(self.chain_id)) if self.persistent else None

    def _deploy(self):
        factory = self.w3.eth.contract(abi=self._abi, bytecode=self._bytecode)
        receipt = self._send(factory.constructor())
        contract = self.w3.eth.contract(address=receipt.contractAddress, abi=self._abi)
        self.deploy_tx = "0x" + _hex(receipt.transactionHash)
        if self.persistent:
            saved = self._read_saved()
            saved[str(self.chain_id)] = contract.address
            ADDRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            ADDRESS_FILE.write_text(json.dumps(saved, indent=2))
        return contract

    def _send(self, fn):
        """Send a transaction (signed locally if a private key is configured)."""
        try:
            if self._account:
                tx = fn.build_transaction(
                    {
                        "from": self.sender,
                        "nonce": self.w3.eth.get_transaction_count(self.sender),
                        "chainId": self.chain_id,
                    }
                )
                signed = self._account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            else:
                tx_hash = fn.transact({"from": self.sender})
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        except Exception as e:
            if "insufficient funds" in str(e).lower():
                raise ChainError(f"Account {self.sender} does not have enough test ETH to pay for this transaction. Get more from a faucet.") from e
            raise
        if receipt.status != 1:
            raise ChainError(f"Transaction 0x{_hex(receipt.transactionHash)} was rejected by the chain.")
        return receipt

    @property
    def address(self) -> str:
        return self.contract.address

    def balance_wei(self) -> int:
        return self.w3.eth.get_balance(self.sender)

    def balance_eth(self) -> float:
        return float(Web3.from_wei(self.balance_wei(), "ether"))

    def explorer_url(self, kind: str, value: str) -> str | None:
        """Link to a block explorer page (`kind` is "tx" or "address"); None on chains without one."""
        base = EXPLORERS.get(self.chain_id)
        return f"{base}/{kind}/{value}" if base else None

    def tx_url(self, tx_hash: str) -> str | None:
        return self.explorer_url("tx", tx_hash)

    def contract_url(self) -> str | None:
        return self.explorer_url("address", self.address)

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

