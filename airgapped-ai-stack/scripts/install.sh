#!/usr/bin/env bash
# Run on the DGX Spark (no internet needed). Loads the bundle and starts the stack.
#
#   ./scripts/install.sh /media/usb/bundle
set -euo pipefail
cd "$(dirname "$0")/.."
BUNDLE=$(cd "${1:?usage: install.sh /path/to/bundle}" && pwd)

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

log "Pre-flight checks"
[[ $(uname -m) == aarch64 ]] || fail "expected an arm64 host (DGX Spark), got $(uname -m)"
command -v docker >/dev/null || fail "docker not found (it ships with DGX OS)"
docker compose version >/dev/null || fail "docker compose plugin not found"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || fail "nvidia-smi failed"
[[ $(docker info 2>/dev/null) == *nvidia* ]] || fail "NVIDIA container runtime not registered with Docker (nvidia-ctk runtime configure --runtime=docker)"

log "Verifying bundle checksums"
(cd "$BUNDLE" && sha256sum -c SHA256SUMS) || fail "bundle is corrupt or was modified"

log "Loading container images"
docker load -i "$BUNDLE/images.tar"

if [[ -f $BUNDLE/ollama-models.tar ]]; then
  log "Installing models into ./data/ollama"
  mkdir -p data/ollama
  tar -C data/ollama -xf "$BUNDLE/ollama-models.tar"
fi
mkdir -p data/open-webui data/chroma

if [[ ! -f .env ]]; then
  log "Creating .env with a fresh WEBUI_SECRET_KEY"
  cp .env.example .env
  sed -i "s|^WEBUI_SECRET_KEY=.*|WEBUI_SECRET_KEY=$(openssl rand -hex 32)|" .env
  chmod 600 .env
fi

set -a; source .env; set +a

log "Checking ports ${WEBUI_PORT} and ${RAG_API_PORT} are free"
for port in "$WEBUI_PORT" "$RAG_API_PORT"; do
  [[ -z $(ss -Hltn "sport = :$port") ]] && continue
  owner=$(docker ps --filter "publish=$port" --format '{{.Names}}')
  [[ $owner == airgap-* ]] && continue   # our own stack from an earlier run
  fail "port $port is already used by ${owner:-a host process (see: sudo ss -ltnp 'sport = :$port')}.
       Pick free ports in .env (e.g. WEBUI_PORT=3300, RAG_API_PORT=8300) and re-run."
done

log "Starting stack (nothing will be pulled or built)"
docker compose up -d --pull never --no-build

log "Waiting for Open WebUI"
for _ in $(seq 1 90); do
  curl -fsS "http://${BIND_ADDR}:${WEBUI_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done

docker compose ps
echo
docker exec airgap-ollama ollama list
echo
echo "Open WebUI : http://${BIND_ADDR}:${WEBUI_PORT}   (first account you create becomes admin)"
echo "RAG API    : http://${BIND_ADDR}:${RAG_API_PORT}/health"
echo "Next       : ./scripts/verify.sh"
