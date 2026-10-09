"""Deploy the CTRRegistry contract to a test network and register the current model.

Usage (Windows Command Prompt):
    set CTR_RPC_URL=https://ethereum-sepolia-rpc.publicnode.com
    set CTR_PRIVATE_KEY=<private key of a throw-away test wallet>
    python deploy_chain.py

Mac/Linux: use `export` instead of `set`.

What it does
1. Connects, prints the network, your account and its test-ETH balance.
2. Deploys the contract (or reuses the one saved for this network).
3. Registers the trained model's fingerprints (dataset, model file, metrics).
4. Anchors any sealed ledger batches that are not on the chain yet.
5. Prints block-explorer links, so you can show the transactions publicly.

It never prints your private key and it refuses Ethereum mainnet.
"""
import json
import sys

import ctr_core as core
from blockchain.chain_client import ChainClient, ChainError
from ledger import Ledger, hash_file


def link(label: str, url: str | None) -> None:
    print(f"  {label}: {url}" if url else f"  {label}: (no block explorer for this network)")


def main() -> int:
    try:
        chain = ChainClient()
    except ChainError as e:
        print(f"\nStopped: {e}")
        return 1

    print(f"Network : {chain.network} (chain id {chain.chain_id})")
    print(f"Account : {chain.sender}")
    print(f"Balance : {chain.balance_eth():.4f} ETH")
    if not chain.persistent:
        print("\nNote: no CTR_RPC_URL is set, so this used a temporary in-memory chain (nothing is saved).")

    print(f"\nContract: {chain.address} ({'reused existing' if chain.reused_contract else 'newly deployed'})")
    if not chain.reused_contract:
        link("deployment transaction", chain.tx_url(chain.deploy_tx))
    link("contract page", chain.contract_url())

    # model provenance
    if core.PROVENANCE_PATH.exists():
        provenance = json.loads(core.PROVENANCE_PATH.read_text())
        current = hash_file(core.MODEL_PATH)
        if current != provenance["model_hash"]:
            print("\nWarning: the model file differs from provenance.json. Run `python main.py` again, then retry.")
        elif chain.find_model(current):
            print("\nModel already registered on this chain.")
        else:
            tx = chain.register_model(provenance)
            print(f"\nModel {provenance['version']} registered.")
            link("transaction", chain.tx_url(tx))
    else:
        print("\nNo provenance.json found: run `python main.py` first to register a model.")

    # ledger batches
    ledger = Ledger()
    anchored = chain.sync_ledger(ledger)
    print(f"\nLedger: {anchored} new batch(es) anchored, {chain.batch_count()} on the chain in total.")

    print(f"\nBalance left: {chain.balance_eth():.4f} ETH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
