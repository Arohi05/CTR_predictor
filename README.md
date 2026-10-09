# CTR Predictor with a blockchain audit trail

Predicts the **click-through rate (CTR = clicks / impressions)** of an ad campaign and keeps a
tamper-proof record of every prediction and of the model itself.

```
Dataset.csv ──► main.py (train) ──► model_output/ ──► app.py (Streamlit UI)
                    │                                   │
             SHA-256 provenance                  prediction records
                    └──────────────┬──────────────────┘
                              ledger.py  (hashes, Merkle trees, hash-chained blocks)
                                   │  block hash + Merkle root
                                   ▼
                      blockchain/CTRRegistry.sol  (smart contract)
```

## Quick start

```bash
pip install -r requirements.txt
python main.py            # trains in a few seconds, writes model_output/
streamlit run app.py      # opens the web app
python -m pytest          # 19 tests: hashing, tamper detection, smart contract
```

No blockchain setup is needed: by default the app uses an in-memory test chain.

## What changed from the first version

| Problem in v1 | Fix |
|---|---|
| Target was the raw `clicks` count, so the classifier had 1,166 "classes" and the app showed the probability of "exactly N clicks" | Target is now `clicks / impressions` (a regression) |
| `media_cost_usd` (known only after the ad ran) was an input: data leakage | Only pre-campaign inputs are used |
| 6 numeric boxes that you had to fill with magic numbers | Dropdowns for platform, channel, market, audience, etc. |
| 9 MB random forest, slow load, 1,166 probabilities per prediction | 0.27 MB gradient boosting model, trains in ~4 s, prediction ~0.1 s |
| App reran everything on every widget change | Form: the model runs only when you click *Predict* |
| Library versions not pinned, model could not be loaded on other versions | `requirements.txt` has minimum versions; the version used is stored in `model_meta.json` |

Honest accuracy (campaigns the model never saw): average error **0.67 percentage points**
vs **1.01** for always guessing the average, R² ≈ 0.50.

## Hashing and blockchain features

* **Record hashing.** Each prediction is a JSON record; its fingerprint is SHA-256 of the
  canonical JSON (`ledger.hash_record`).
* **Merkle tree.** A batch of records is summarised by one Merkle root. Any single record can be
  proven to belong to the batch with a short proof (*Prove a single prediction* in the app).
* **Hash chain.** Each batch is a block holding the previous block's hash. Editing any old record
  breaks every later block, and `Ledger.verify()` points to the first broken one.
* **Smart contract** (`blockchain/CTRRegistry.sol`). Stores each batch's block hash and Merkle
  root, and registers each trained model (dataset hash, model-file hash, metrics hash, version).
  Only the owner can write, nothing can be edited or deleted, and raw data is never put on-chain.
* **Model provenance.** `main.py` writes `model_output/provenance.json`. The app checks that the
  model file on disk still matches, and whether it is registered on-chain.

Why this matters: an attacker (or the app owner) can rebuild a perfectly consistent fake ledger,
but it cannot match what is anchored on the blockchain (see `tests/test_chain.py`,
`test_rewriting_history_is_caught_by_the_chain`).

> SHA-256 is used for off-chain data. The contract only stores the resulting 32-byte values.

## Using a real chain

```bash
# local node (separate terminal)
npx hardhat node                       # or: npx ganache
export CTR_RPC_URL=http://127.0.0.1:8545
streamlit run app.py                   # deploys the contract and remembers its address

# public testnet (e.g. Sepolia): use a throw-away account funded from a faucet
export CTR_RPC_URL=<your node url>
export CTR_PRIVATE_KEY=<test account key>      # never commit it, never use a real-money key
```

Set `CTR_CONTRACT_ADDRESS` to reuse an existing deployment.

To change the contract: `npm install solc@0.8.24 && node blockchain/compile.js`.

## Files

| File | Purpose |
|---|---|
| `main.py` | Train the model, save metrics and provenance hashes |
| `ctr_core.py` | Feature list and data cleaning shared by training and the app |
| `app.py` | Streamlit frontend (Predict, Ledger & blockchain, Model tabs) |
| `ledger.py` | SHA-256 hashing, Merkle tree and proofs, hash-chained ledger |
| `blockchain/` | Solidity contract, compiled ABI, Python web3 client |
| `tests/` | Automated tests |
| `Main.ipynb` | Original exploration notebook (kept for reference; `main.py` supersedes its model) |
