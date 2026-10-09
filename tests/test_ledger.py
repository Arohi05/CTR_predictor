import json

import pytest

from ledger import Ledger, hash_record, merkle_proof, merkle_root, verify_proof


def rec(i):
    return {"campaign": i, "ctr": 0.01 * i, "channel": "Mobile"}


def test_hash_is_order_independent_and_deterministic():
    assert hash_record({"a": 1, "b": 2}) == hash_record({"b": 2, "a": 1})
    assert hash_record({"a": 1}) != hash_record({"a": 2})


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_merkle_proofs_verify_for_every_leaf(n):
    leaves = [hash_record(rec(i)) for i in range(n)]
    root = merkle_root(leaves)
    for i in range(n):
        assert verify_proof(leaves[i], merkle_proof(leaves, i), root)


def test_merkle_proof_fails_for_wrong_leaf():
    leaves = [hash_record(rec(i)) for i in range(6)]
    root = merkle_root(leaves)
    assert not verify_proof(hash_record(rec(99)), merkle_proof(leaves, 2), root)


def make_ledger(tmp_path, batches=3, per_batch=4):
    led = Ledger(tmp_path)
    for b in range(batches):
        for i in range(per_batch):
            led.add_record(rec(b * 10 + i))
        led.seal_batch()
    return led


def test_valid_chain_verifies(tmp_path):
    led = make_ledger(tmp_path)
    assert len(led.blocks()) == 3
    assert led.verify()["ok"]


def test_empty_pending_does_not_seal(tmp_path):
    assert Ledger(tmp_path).seal_batch() is None


def test_changing_a_record_is_detected(tmp_path):
    led = make_ledger(tmp_path)
    path = tmp_path / "batch_1.json"
    data = json.loads(path.read_text())
    data[2]["ctr"] = 0.99  # attacker edits one stored prediction
    path.write_text(json.dumps(data))
    result = led.verify()
    assert not result["ok"] and result["block"] == 1


def test_changing_a_block_header_is_detected(tmp_path):
    led = make_ledger(tmp_path)
    chain = json.loads(led.chain_path.read_text())
    chain[0]["timestamp"] += 1
    led.chain_path.write_text(json.dumps(chain))
    assert not led.verify()["ok"]


def test_removing_a_block_is_detected(tmp_path):
    led = make_ledger(tmp_path)
    chain = json.loads(led.chain_path.read_text())
    del chain[1]
    led.chain_path.write_text(json.dumps(chain))
    assert not led.verify()["ok"]


def test_prove_record(tmp_path):
    led = make_ledger(tmp_path)
    p = led.prove_record(2, 3)
    assert verify_proof(p["leaf"], p["proof"], p["root"])
