#!/usr/bin/env bash
# Manage the Fabric test-network for ECG-Trace.
#
#   ./scripts/fabric.sh up       # (re)start network + create channel + deploy chaincode
#   ./scripts/fabric.sh deploy   # (re)deploy chaincode only — bump CC_SEQUENCE for an upgrade
#   ./scripts/fabric.sh down     # stop everything (the ledger is erased!)
#
# Env: FABRIC_SAMPLES (default ~/fabric-samples), CHANNEL (ecgchannel), CC_NAME (ecgtrace),
#      CC_VERSION (1.0), CC_SEQUENCE (1)
set -euo pipefail

FABRIC_SAMPLES=${FABRIC_SAMPLES:-$HOME/fabric-samples}
CHANNEL=${CHANNEL:-ecgchannel}
CC_NAME=${CC_NAME:-ecgtrace}
CC_VERSION=${CC_VERSION:-1.0}
CC_SEQUENCE=${CC_SEQUENCE:-1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NET="$FABRIC_SAMPLES/test-network"

[ -x "$NET/network.sh" ] || { echo "test-network not found in $NET (set FABRIC_SAMPLES)"; exit 1; }
export PATH="$FABRIC_SAMPLES/bin:$PATH"

deploy() {
  # package a clean copy: no node_modules, no tests
  local tmp; tmp=$(mktemp -d)
  cp -r "$REPO/chaincode/ecg-trace/." "$tmp/"
  rm -rf "$tmp/node_modules" "$tmp/test"
  (cd "$NET" && ./network.sh deployCC -c "$CHANNEL" -ccn "$CC_NAME" -ccp "$tmp" -ccl javascript \
      -ccv "$CC_VERSION" -ccs "$CC_SEQUENCE")
  rm -rf "$tmp"
  echo "Chaincode $CC_NAME v$CC_VERSION (sequence $CC_SEQUENCE) deployed on channel $CHANNEL"
}

case "${1:-up}" in
  up)
    (cd "$NET" && ./network.sh down && ./network.sh up createChannel -c "$CHANNEL")
    deploy
    echo
    echo "Next: cd blockchain-gateway && npm install && npm start"
    ;;
  deploy) deploy ;;
  down)   (cd "$NET" && ./network.sh down) ;;
  *) echo "usage: $0 up|deploy|down"; exit 1 ;;
esac