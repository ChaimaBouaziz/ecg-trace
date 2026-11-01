# Hyperledger Fabric setup (Windows + WSL2 + Docker Desktop)

Everything below runs in the **Ubuntu (WSL) terminal**, never in PowerShell.

```
Python pipeline ──HTTP──> blockchain-gateway (Node, :3001) ──gRPC──> peer0.org1 ──> chaincode "ecgtrace"
  (ledger/fabric_client.py)    (blockchain-gateway/server.js)        test-network      (chaincode/ecg-trace)
```

## 1. Prerequisites (once)

```bash
# Docker: Docker Desktop -> Settings -> Resources -> WSL integration -> enable Ubuntu
docker ps                        # must work inside WSL (empty list is fine)

sudo apt update && sudo apt install -y curl git jq

# Node.js 20 via nvm (chaincode tests + gateway)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install 20
node -v                          # v20.x
```

## 2. Install Fabric samples, binaries and Docker images (once, ~2 GB)

```bash
cd ~
curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh
chmod +x install-fabric.sh
./install-fabric.sh docker samples binary
ls ~/fabric-samples/test-network/network.sh     # must exist
```

Optional sanity check with the official sample:
```bash
cd ~/fabric-samples/test-network
./network.sh up createChannel && ./network.sh down
```

## 3. Test the chaincode logic (no network needed)

```bash
cd ~/ecg-trace/chaincode/ecg-trace
npm install
npm test                          # 6 tests pass
```

## 4. Start the network + deploy the ECG-Trace chaincode

```bash
cd ~/ecg-trace
chmod +x scripts/fabric.sh
./scripts/fabric.sh up            # 2 orgs, 1 orderer, channel "ecgchannel", chaincode "ecgtrace"
docker ps                         # peer0.org1, peer0.org2, orderer + 2 chaincode containers
```
The first deployment takes a few minutes (Docker builds the Node chaincode image).

## 5. Start the blockchain gateway (keep this terminal open)

```bash
cd ~/ecg-trace/blockchain-gateway
npm install
npm start                         # "Blockchain gateway on http://localhost:3001 ..."
```
Check from another terminal: `curl localhost:3001/health`

## 6. Run the pipeline on Fabric

```bash
cd ~/ecg-trace && source .venv/bin/activate
LEDGER=fabric python scripts/run_case_demo.py --tamper
```
Each artifact is now a real Fabric transaction (`tx=` ids), and the integrity check compares
the off-chain files with the hashes stored on the blockchain.

Useful queries:
```bash
curl localhost:3001/cases/<CASE-ID>/history | jq
curl localhost:3001/artifacts/<ARTIFACT-ID>/key-history | jq    # written once, never modified
```

## What the chaincode enforces

| Rule | Why |
|---|---|
| Artifact ids can be written only once | append-only provenance |
| `sha256` = 64 hex chars, known `artifact_type`, required fields | data quality on-chain |
| Parents must exist and belong to the same case | the chain cannot be faked |
| Only `Org1MSP` may write | "only authorised services may submit transactions" |
| `tx_id`, `ledger_timestamp`, `submitter` set by the chaincode | cannot be forged by the client |

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker: command not found` in WSL | enable WSL integration in Docker Desktop, restart the terminal |
| `$'\r': command not found` | file has Windows line endings: `sed -i 's/\r$//' scripts/fabric.sh` |
| `peer: command not found` | `export PATH=$HOME/fabric-samples/bin:$PATH` |
| Gateway: `ENOENT ... ca.crt` | network not started, or set `FABRIC_SAMPLES=/path/to/fabric-samples` |
| Gateway: `DEADLINE_EXCEEDED` / `UNAVAILABLE` | network stopped (e.g. after a reboot) -> `./scripts/fabric.sh up` |
| After `fabric.sh down/up` old cases are UNVERIFIABLE | expected: the test-network ledger is erased on `down` |
| Changed the chaincode | `CC_VERSION=1.1 CC_SEQUENCE=2 ./scripts/fabric.sh deploy` |