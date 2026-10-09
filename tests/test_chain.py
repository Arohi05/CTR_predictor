import pytest

from blockchain.chain_client import ChainClient
from ledger import Ledger, hash_record


@pytest.fixture()
def client():
    return ChainClient()  # in-memory test chain


def filled_ledger(tmp_path, batches=2):
    led = Ledger(tmp_path)
    for b in range(batches):
        for i in range(3):
            led.add_record({"b": b, "i": i})
        led.seal_batch()
    return led


def test_sync_anchors_all_blocks_once(client, tmp_path):
    led = filled_ledger(tmp_path)
    assert client.sync_ledger(led) == 2
    assert client.batch_count() == 2
    assert client.sync_ledger(led) == 0  # idempotent
    assert all(r["status"] == "matches chain" for r in client.verify_ledger(led))


def test_block_stored_on_chain_equals_local_block(client, tmp_path):
    led = filled_ledger(tmp_path, 1)
    client.sync_ledger(led)
    stored, local = client.get_batch(0), led.blocks()[0]
    assert stored["block_hash"] == local.block_hash
    assert stored["merkle_root"] == local.merkle_root
    assert stored["record_count"] == 3


def test_rewriting_history_is_caught_by_the_chain(client, tmp_path):
    """Even if an attacker rebuilds a consistent fake ledger, the chain disagrees."""
    led = filled_ledger(tmp_path / "real")
    client.sync_ledger(led)
    fake = Ledger(tmp_path / "fake")
    for b in range(2):
        for i in range(3):
            fake.add_record({"b": b, "i": i + (100 if b == 0 else 0)})  # altered batch 0
        fake.seal_batch()
    assert fake.verify()["ok"]  # internally consistent...
    statuses = [r["status"] for r in client.verify_ledger(fake)]
    assert statuses[0] == "MISMATCH with chain"  # ...but not what was anchored


def test_only_owner_can_write(client):
    other = client.w3.eth.accounts[1]
    with pytest.raises(Exception):
        client.contract.functions.anchorBatch(b"\x00" * 32, b"\x01" * 32, 1).transact({"from": other})


def test_model_registration_and_lookup(client):
    prov = {
        "version": "v1",
        "model_hash": hash_record({"m": 1}),
        "dataset_hash": hash_record({"d": 1}),
        "metrics_hash": hash_record({"x": 1}),
    }
    assert client.find_model(prov["model_hash"]) is None
    client.register_model(prov)
    found = client.find_model(prov["model_hash"])
    assert found["version"] == "v1" and found["dataset_hash"] == prov["dataset_hash"]
    assert client.find_model(hash_record({"m": 2})) is None


# ---------------------------------------------------------------------------
# Testnet path: transactions signed locally with a private key (what Sepolia uses)
# ---------------------------------------------------------------------------
from types import SimpleNamespace

from web3 import EthereumTesterProvider, Web3

from blockchain.chain_client import ChainError


def funded_key_and_w3():
    provider = EthereumTesterProvider()
    w3 = Web3(provider)
    key = provider.ethereum_tester.backend.account_keys[0].to_hex()  # a pre-funded test account
    return key, w3


@pytest.mark.parametrize("strip_prefix", [False, True])
def test_signed_transactions_work_with_and_without_0x_prefix(tmp_path, strip_prefix):
    key, w3 = funded_key_and_w3()
    client = ChainClient(w3=w3, private_key=key[2:] if strip_prefix else key)
    led = filled_ledger(tmp_path, 2)
    assert client.sync_ledger(led) == 2
    assert all(r["status"] == "matches chain" for r in client.verify_ledger(led))
    prov = {"version": "v1", "model_hash": "ab" * 32, "dataset_hash": "cd" * 32, "metrics_hash": "ef" * 32}
    client.register_model(prov)
    assert client.find_model(prov["model_hash"])["version"] == "v1"
    assert client.balance_eth() > 0


def test_mainnet_is_refused():
    fake = SimpleNamespace(eth=SimpleNamespace(chain_id=1))
    with pytest.raises(ChainError, match="mainnet"):
        ChainClient(w3=fake)


def test_bad_private_key_gives_a_clear_error_without_leaking_it():
    _, w3 = funded_key_and_w3()
    with pytest.raises(ChainError) as err:
        ChainClient(w3=w3, private_key="not-a-key-123")
    assert "not-a-key-123" not in str(err.value)


def test_explorer_links_only_on_known_public_networks(client):
    assert client.tx_url("0x" + "00" * 32) is None  # local test chain has no explorer
    client.chain_id = 11155111
    assert client.tx_url("0xabc") == "https://sepolia.etherscan.io/tx/0xabc"
    assert client.contract_url().startswith("https://sepolia.etherscan.io/address/0x")
