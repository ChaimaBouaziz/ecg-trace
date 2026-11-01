# ECG-Trace — Agentic, traceable ECG analysis (AI + Hyperledger Fabric)

Mini-project 2026. Pipeline: **raw ECG + report → preprocessing → AI prediction → validation →
explanation → final result**, every artifact hashed (SHA-256) and its fingerprint + provenance
recorded on Hyperledger Fabric. Dataset: **PTB-XL v1.0.3** (PhysioNet).

> Educational prototype — not a medical diagnostic tool.

## Roadmap

| Step | Content | Owner | Status |
|---|---|---|---|
| 0 | Repo, env, dataset, shared contract | all | ✅ |
| 1 | ECG preprocessing (filter, normalise, segment) — `pipeline/preprocess.py`, `scripts/run_preprocess.py` | A | ✅ |
| 2 | AI model (NORM vs ABNORMAL, versioned) — `pipeline/features.py`, `pipeline/model.py`, `scripts/train_model.py` | A | ✅ |
| 3 | Validation rules + explanation + final result — `pipeline/validate.py`, `pipeline/explain.py`, `scripts/evaluate_validation.py` | A | ✅ |
| 4 | Artifact chain + hashing — `pipeline/artifacts.py`, `pipeline/storage.py` | C | ✅ |
| 5 | End-to-end with FakeLedger + integrity check — `pipeline/run_case.py`, `pipeline/verify.py`, `scripts/run_case_demo.py` | C | ✅ |
| 6 | Fabric: chaincode `chaincode/ecg-trace`, REST gateway `blockchain-gateway`, `ledger/fabric_client.py` — see `docs/FABRIC_SETUP.md` | B | ✅ (to run on your machine) |
| 7 | Agents: LangGraph orchestrator, 7 agents + audit, retries, traced errors, pseudonymisation — `agents/orchestrator.py`, `scripts/run_agents_demo.py` | C | ✅ |
| 8 | FastAPI + web dashboard (live agents, ECG chart, provenance, verify, tamper demo) — `api/main.py`, `ui/index.html` | C | ✅ |
| 9 | Integrity verification + tamper demo + tests | all | |
| 10 | Docker (`Dockerfile`, `blockchain-gateway/Dockerfile`, `docker-compose.yml`, `scripts/demo_up.sh`) + final report | all | Docker ✅ |

## The shared contract (read this first)

Everyone codes against these three files — change them only after agreeing as a team:

- `pipeline/schemas.py` — `ArtifactRecord` = exactly what goes on-chain (spec §4.6):
  `artifact_id, case_id, artifact_type, sha256, created_at, software_version, parent_ids,
  offchain_uri, creator_id, tx_id`. Medical data itself **never** goes on-chain.
- `pipeline/hashing.py` — canonical hashing (`sha256_json`, `sha256_array`, `sha256_text`,
  `sha256_file`). Always hash through these so hashes are reproducible.
- `ledger/base.py` — `LedgerClient` interface (`put_artifact`, `get_artifact`,
  `get_case_history`). `FakeLedger` implements it now; `FabricLedger` (step 6) will implement
  the same methods → swapping is one line.

Artifact chain: `RAW_ECG + RAW_REPORT → PROCESSED → FEATURES → PREDICTION → VALIDATION →
EXPLANATION → FINAL` (each record lists its `parent_ids`).

## Setup (Windows + WSL2 + Docker Desktop)

1. Docker Desktop → *Settings → Resources → WSL integration* → enable your Ubuntu distro.
2. Work **inside the Linux filesystem** (`~/…`), not under `/mnt/c/…` (much faster, and Fabric
   scripts break on Windows paths/line endings).

```bash
# in the Ubuntu (WSL) terminal
sudo apt update && sudo apt install -y python3-venv python3-pip git
cd ~ && git clone <your-repo-url> ecg-trace && cd ecg-trace
git config core.autocrlf input          # avoid CRLF issues
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # 4 tests should pass
```

## Get the data

```bash
python scripts/download_ptbxl.py --n 2000      # fast: CSVs + 2000 records @100 Hz (~few min)
# or: python scripts/download_ptbxl.py --mode full   # all 21,799 records (~1.7 GB zip)
python scripts/explore_ptbxl.py                # stats + figures in docs/exploration/
```

PTB-XL facts used later:
- 10 s, 12 leads, 100 Hz → signal shape `(1000, 12)`
- labels: `scp_codes` → 5 diagnostic superclasses `NORM, MI, STTC, CD, HYP`
- `report` column = the medical report (often German) → ECG↔report coherence check
- `baseline_drift, static_noise, burst_noise, electrodes_problems…` → data-quality check
- `strat_fold`: folds 1–8 train, 9 val, 10 test (official split; records of a patient stay together)
- data is already pseudonymised; our pipeline still assigns its own `CASE-…` id

## Run the demo (3 terminals, all in WSL)

```bash
./scripts/fabric.sh up                                   # 1. Fabric network + chaincode
cd blockchain-gateway && npm start                       # 2. gateway (keep open)
source .venv/bin/activate
LEDGER=fabric uvicorn api.main:app --port 8000           # 3. API + dashboard (keep open)
```
Open http://localhost:8000 -> run a sample -> watch the agents -> **Verify integrity** ->
**Tamper file** -> **Verify integrity** (MODIFIED) -> **Restore** -> **Verify** (VALID).
Without Fabric: `uvicorn api.main:app --port 8000` (FakeLedger). API docs: http://localhost:8000/docs.
Optional security: `API_KEY=secret` -> write operations require the `X-API-Key` header.

## Run with Docker (one command)

```bash
./scripts/demo_up.sh          # starts Fabric if needed, then builds + runs gateway and API containers
./scripts/demo_up.sh down     # stops the app containers (Fabric keeps running)
```
- Fabric runs in the official test-network containers; the app containers join its `fabric_test` network.
- `data/` is mounted from your disk, so off-chain files and PTB-XL samples persist between runs.
- Stop the manual `npm start` / `uvicorn` first (ports 3001 and 8000).
- FakeLedger only: `docker network create fabric_test` once, then `LEDGER=fake docker compose up -d --build api`.

## Layout

```
pipeline/   schemas, hashing, ptbxl loader, (next) preprocess, model, validate
ledger/     base interface, fake_ledger, fabric_client, factory (LEDGER=fake|fabric)
chaincode/  Fabric chaincode (JavaScript)
blockchain-gateway/  Node REST gateway Python -> Fabric
agents/     LangGraph orchestrator + specialised agents
api/        FastAPI backend (upload, jobs, cases, verify, demo tamper/restore)
ui/         single-page dashboard served at http://localhost:8000
scripts/    download / explore / train
data/       raw/ (git-ignored), offchain/ (git-ignored)
tests/
```