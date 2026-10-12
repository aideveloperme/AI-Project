#!/usr/bin/env bash
# Checks the running stack: services up, GPU in use, and containers can't reach out.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

pass=0; failc=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$*"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; failc=$((failc+1)); }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }

# Try a TCP connection from inside a container using bash's /dev/tcp.
can_reach() { docker exec "$1" timeout 5 bash -c "exec 3<>/dev/tcp/$2/$3" >/dev/null 2>&1; }

echo "Services"
for c in airgap-ollama airgap-chromadb airgap-rag-api airgap-open-webui; do
  [[ $(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null) == true ]] && ok "$c running" || bad "$c not running"
done
curl -fsS "http://${BIND_ADDR}:${WEBUI_PORT}/health" >/dev/null && ok "Open WebUI answers" || bad "Open WebUI not answering"
h=$(curl -fsS "http://${BIND_ADDR}:${RAG_API_PORT}/health") && echo "$h" | grep -q '"chroma":{"ok":true' \
  && echo "$h" | grep -q '"ollama":{"ok":true' && ok "rag-api sees Ollama + Chroma" || bad "rag-api health: ${h:-no response}"

echo "GPU"
docker exec airgap-ollama nvidia-smi -L >/dev/null 2>&1 && ok "GPU visible in ollama container" || bad "no GPU in ollama container"
model=$(docker exec airgap-ollama ollama list | awk 'NR>1 && $1 !~ /embed/ {print $1; exit}')
if [[ -n $model ]]; then
  docker exec airgap-ollama ollama run "$model" "Reply with OK" >/dev/null 2>&1
  docker exec airgap-ollama ollama ps | grep -q GPU && ok "$model runs on GPU" || bad "$model not on GPU: $(docker exec airgap-ollama ollama ps | tail -1)"
else
  bad "no chat model installed"
fi

echo "Isolation"
for c in airgap-ollama airgap-chromadb; do
  if ! docker exec "$c" bash -c true 2>/dev/null; then warn "$c has no bash; skipped"; continue; fi
  can_reach "$c" 1.1.1.1 443 && bad "$c can reach the internet" || ok "$c has no outbound route (internal network)"
done
for c in airgap-rag-api airgap-open-webui; do
  can_reach "$c" 1.1.1.1 443 \
    && warn "$c can reach 1.1.1.1 — host itself is online; disconnect it or firewall it" \
    || ok "$c cannot reach the internet"
done
docker exec airgap-open-webui bash -c 'getent hosts huggingface.co' >/dev/null 2>&1 \
  && warn "DNS resolves public names (host has a resolver configured)" || ok "no public DNS"

echo
echo "$pass passed, $failc failed"
[[ $failc -eq 0 ]]
