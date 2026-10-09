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
