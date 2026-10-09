"""CTR Predictor: Streamlit frontend with a blockchain-backed audit trail.

Run:  streamlit run app.py
"""
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st

import ctr_core as core
from ledger import Ledger, hash_file, verify_proof

st.set_page_config(page_title="CTR Predictor", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(core.MODEL_PATH), core.load_meta()


@st.cache_resource(show_spinner="Connecting to blockchain...")
def get_chain():
    from blockchain.chain_client import ChainClient

    try:
        return ChainClient(), None
    except Exception as e:  # node unreachable, bad key, ...
        return None, str(e)


def short(h: str) -> str:
    return f"{h[:10]}...{h[-6:]}" if h else ""


if not core.MODEL_PATH.exists() or not core.META_PATH.exists():
    st.error("No trained model found. Run `python main.py` once, then reload this page.")
    st.stop()

pipe, meta = load_model()
chain, chain_error = get_chain()
ledger = Ledger()
provenance = json.loads(core.PROVENANCE_PATH.read_text()) if core.PROVENANCE_PATH.exists() else None

# ------------------------------------------------------------------- header
st.title("Click-Through Rate Predictor")
st.caption("Predict the CTR of an ad campaign, and keep a tamper-proof record of every prediction on a blockchain.")

tab_predict, tab_chain, tab_model = st.tabs(["Predict", "Ledger & blockchain", "Model"])

# ------------------------------------------------------------------ predict
with tab_predict:
    d, cats = meta["defaults"], meta["categories"]
    with st.form("predict_form"):
        st.subheader("Campaign details")
        c1, c2, c3 = st.columns(3)
        values = {}
        labels = {
            "ext_service_name": "Ad platform",
            "channel_name": "Channel",
            "advertiser_name": "Advertiser / market",
            "advertiser_currency": "Currency",
            "search_tag_cat": "Audience type",
            "creative_size": "Creative size",
            "template_id": "Template ID",
            "timezone": "Timezone",
            "weekday_cat": "Day type",
        }
        for i, feat in enumerate(core.CATEGORICAL):
            col = (c1, c2, c3)[i % 3]
            options = cats[feat]
            values[feat] = col.selectbox(labels[feat], options, index=options.index(d[feat]))
        n1, n2, n3 = st.columns(3)
        lo, hi = meta["numeric_ranges"]["campaign_day"]
        values["campaign_day"] = n1.number_input("Day of the campaign", min_value=1, max_value=max(hi, 400), value=d["campaign_day"], step=1)
        values["campaign_budget_usd"] = n2.number_input("Campaign budget (USD)", min_value=0.0, value=float(d["campaign_budget_usd"]), step=50.0)
        planned_impressions = n3.number_input("Planned impressions (optional)", min_value=0, value=10000, step=1000)
        log_it = st.checkbox("Record this prediction in the tamper-proof ledger", value=True)
        submitted = st.form_submit_button("Predict CTR", type="primary")

    if submitted:
        ctr = float(pipe.predict(core.make_input_row(values))[0])
        ctr = max(ctr, 0.0)
        st.session_state["last"] = {"values": values, "ctr": ctr, "impressions": planned_impressions}
        if log_it:
            record = {
                "inputs": {k: (v if isinstance(v, str) else float(v)) for k, v in values.items()},
                "predicted_ctr": round(ctr, 6),
                "model_version": provenance["version"] if provenance else "unknown",
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            st.session_state["last"]["record_hash"] = ledger.add_record(record)

    if "last" in st.session_state:
        last = st.session_state["last"]
        avg = meta["average_ctr"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted CTR", f"{last['ctr'] * 100:.2f}%", f"{(last['ctr'] - avg) * 100:+.2f} pts vs average", delta_color="normal")
        m2.metric("Dataset average CTR", f"{avg * 100:.2f}%")
        m3.metric("Expected clicks", f"{core.expected_clicks(last['ctr'], last['impressions']):,.0f}", f"per {last['impressions']:,} impressions", delta_color="off")
        if "record_hash" in last:
            st.success(f"Recorded in ledger. Record fingerprint (SHA-256): `{last['record_hash']}`")
            st.caption("Seal the batch in the *Ledger & blockchain* tab to anchor it on the chain.")

# ------------------------------------------------------------------- chain
with tab_chain:
    if chain is None:
        st.error(f"Blockchain not available: {chain_error}")
    else:
        contract_link = chain.contract_url()
        st.write(
            f"**Network:** {chain.network} (chain id {chain.chain_id})  |  "
            f"**Contract:** " + (f"[`{chain.address}`]({contract_link})" if contract_link else f"`{chain.address}`")
        )
        if chain.persistent:
            st.caption(f"Account `{chain.sender}` has {chain.balance_eth():.4f} test ETH. Every anchoring costs a small fee in test ETH.")
        else:
            st.info("Using an in-memory test chain: it resets when the app restarts, and the ledger is re-anchored automatically. "
                    "Set `CTR_RPC_URL` and `CTR_PRIVATE_KEY` to use a real test network such as Sepolia (see README).")
            if chain.batch_count() < len(ledger.blocks()):
                chain.sync_ledger(ledger)  # re-anchor after a restart of the test chain
        missing = len(ledger.blocks()) - chain.batch_count()
        if chain.persistent and missing > 0:
            st.warning(f"{missing} sealed batch(es) are not on the chain yet.")
            if st.button("Anchor the missing batches"):
                try:
                    chain.sync_ledger(ledger)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not anchor: {e}")

    st.subheader("1. Prediction ledger")
    pending = ledger.pending()
    p1, p2 = st.columns([1, 3])
    p1.metric("Pending records", len(pending))
    if p2.button("Seal batch and anchor on blockchain", type="primary", disabled=not pending):
        block = ledger.seal_batch()
        if chain is not None:
            try:
                tx = chain.anchor_block(block)
                st.session_state["flash"] = (
                    "success",
                    f"Batch {block.index} sealed ({block.record_count} records) and anchored. Transaction: "
                    + (f"[`{tx}`]({chain.tx_url(tx)})" if chain.tx_url(tx) else f"`{tx}`"),
                )
            except Exception as e:  # e.g. out of test ETH: the batch stays sealed and can be anchored later
                st.session_state["flash"] = ("error", f"Batch {block.index} was sealed locally but could not be anchored: {e}")
        else:
            st.session_state["flash"] = ("warning", f"Batch {block.index} sealed locally; blockchain unavailable, so it was not anchored.")
        time.sleep(0.2)
        st.rerun()
    if "flash" in st.session_state:  # shown once, after the page reloads
        kind, text = st.session_state.pop("flash")
        getattr(st, kind)(text)

    blocks = ledger.blocks()
    local = ledger.verify()
    if blocks:
        statuses = {r["index"]: r["status"] for r in chain.verify_ledger(ledger)} if chain else {}
        st.dataframe(
            pd.DataFrame(
                {
                    "Batch": [b.index for b in blocks],
                    "Records": [b.record_count for b in blocks],
                    "Time (UTC)": [datetime.fromtimestamp(b.timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") for b in blocks],
                    "Merkle root": [short(b.merkle_root) for b in blocks],
                    "Block hash": [short(b.block_hash) for b in blocks],
                    "Previous hash": [short(b.prev_hash) for b in blocks],
                    "On-chain check": [statuses.get(b.index, "n/a") for b in blocks],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if local["ok"]:
            st.success("Ledger integrity check passed: every record, Merkle root and block link recomputes correctly.")
        else:
            st.error(f"TAMPERING DETECTED in batch {local['block']}: {local['reason']}")

        st.subheader("2. Prove a single prediction")
        s1, s2 = st.columns(2)
        b_idx = s1.selectbox("Batch", [b.index for b in blocks])
        records = ledger.batch_records(b_idx)
        r_idx = s2.number_input("Record number", min_value=0, max_value=max(len(records) - 1, 0), value=0, step=1)
        if records:
            with st.expander("Show record"):
                st.json(records[int(r_idx)])
            proof = ledger.prove_record(b_idx, int(r_idx))
            valid = verify_proof(proof["leaf"], proof["proof"], proof["root"])
            st.write(f"Record fingerprint: `{proof['leaf']}`")
            st.write(f"Merkle proof has **{len(proof['proof'])}** steps, so this record can be proven to belong to the anchored root without revealing the other records.")
            (st.success if valid else st.error)("Proof valid: record belongs to the batch." if valid else "Proof INVALID.")
    else:
        st.write("No batches yet. Make a prediction in the *Predict* tab, then seal a batch here.")

    st.subheader("3. Model provenance")
    if provenance is None:
        st.write("Run `python main.py` to create provenance hashes.")
    else:
        current_hash = hash_file(core.MODEL_PATH)
        st.write(f"Version **{provenance['version']}**")
        st.code(
            f"dataset  {provenance['dataset_hash']}\nmodel    {provenance['model_hash']}\nmetrics  {provenance['metrics_hash']}",
            language="text",
        )
        if current_hash == provenance["model_hash"]:
            st.success("Model file on disk matches the hash recorded at training time.")
        else:
            st.error("Model file on disk has CHANGED since it was trained.")
        if chain is not None:
            found = chain.find_model(current_hash)
            if found:
                st.success(f"Registered on-chain as model #{found['index']} (version {found['version']}).")
            else:
                st.warning("This model is not registered on the blockchain yet.")
                if st.button("Register model on blockchain"):
                    try:
                        tx = chain.register_model(provenance)
                        st.session_state["flash"] = (
                            "success",
                            "Model registered. Transaction: " + (f"[`{tx}`]({chain.tx_url(tx)})" if chain.tx_url(tx) else f"`{tx}`"),
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not register the model: {e}")

# -------------------------------------------------------------------- model
with tab_model:
    m = meta["metrics"]
    st.subheader("How good is the model?")
    a, b, c = st.columns(3)
    a.metric("Average error (MAE)", f"{m['mae'] * 100:.2f} pts", f"baseline {m['baseline_mae'] * 100:.2f} pts", delta_color="off")
    b.metric("R²", f"{m['r2']:.2f}")
    c.metric("Training rows", f"{m['train_rows']:,}")
    st.write(
        "The test score comes from **campaigns the model never saw during training**, so it is an honest estimate for new campaigns. "
        "A guess that always says the average CTR is off by "
        f"{m['baseline_mae'] * 100:.2f} percentage points; the model is off by {m['mae'] * 100:.2f}."
    )
    st.write(f"Model: gradient-boosted trees (scikit-learn {meta['sklearn_version']}), trained {meta['trained_at']}.")
    st.write("Inputs used: " + ", ".join(f"`{f}`" for f in meta["features"]))
    st.caption("Inputs deliberately excluded because they are only known after an ad runs: impressions, clicks, media cost, reach.")
