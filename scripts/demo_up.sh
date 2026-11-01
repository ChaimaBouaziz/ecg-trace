#!/usr/bin/env bash
# One command for the full demo: Fabric network (if not running) + gateway + API/dashboard in Docker.
#   ./scripts/demo_up.sh          start everything
#   ./scripts/demo_up.sh down     stop the app containers (Fabric keeps running)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export FABRIC_SAMPLES=${FABRIC_SAMPLES:-$HOME/fabric-samples}

if [ "${1:-up}" = "down" ]; then
  docker compose down
  exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -qx 'peer0.org1.example.com'; then
  echo ">> Fabric network not running: starting it (ledger starts empty)"
  ./scripts/fabric.sh up
else
  echo ">> Fabric network already running"
fi

docker compose up -d --build
echo ">> Waiting for the API..."
for i in $(seq 1 30); do
  if curl -fs localhost:8000/api/health > /dev/null; then break; fi
  sleep 2
done
curl -s localhost:8000/api/health | head -c 300; echo
echo
echo "Dashboard : http://localhost:8000"
echo "API docs  : http://localhost:8000/docs"
echo "Logs      : docker compose logs -f"