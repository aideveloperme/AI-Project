#!/usr/bin/env bash
# Run on a machine WITH internet access (any Linux/macOS box with Docker, or the
# Spark itself before you disconnect it). Produces ./bundle/ — copy that folder
# to removable media and carry it across the air gap.
#
#   ./scripts/bundle.sh                 # images + models from models.txt
#   SKIP_MODELS=1 ./scripts/bundle.sh   # images only (e.g. version bump)
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE=.env; [[ -f $ENV_FILE ]] || ENV_FILE=.env.example
set -a; source "$ENV_FILE"; set +a

PLATFORM=linux/arm64          # DGX Spark = Grace (Arm) CPU + Blackwell GPU
OUT=bundle
MODEL_VOLUME=airgap-bundle-ollama
IMAGES=(
  "ollama/ollama:${OLLAMA_VERSION}"
  "ghcr.io/open-webui/open-webui:${OPEN_WEBUI_VERSION}"
  "chromadb/chroma:${CHROMA_VERSION}"
  "airgap/rag-api:local"
)

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
mkdir -p "$OUT"

# 1. Models. GGUF blobs are architecture-independent, so this runs Ollama
#    natively on whatever machine you are on. Done before pulling the arm64
#    images so the native Ollama image doesn't replace the arm64 tag locally.
if [[ -z ${SKIP_MODELS:-} ]]; then
  mapfile -t MODELS < <(sed -e 's/#.*//' -e 's/[[:space:]]//g' models.txt | grep -v '^$')
  log "Downloading ${#MODELS[@]} models: ${MODELS[*]}"
  docker rm -f airgap-bundler >/dev/null 2>&1 || true
  docker run -d --name airgap-bundler -v "$MODEL_VOLUME":/root/.ollama "ollama/ollama:${OLLAMA_VERSION}" >/dev/null
  trap 'docker rm -f airgap-bundler >/dev/null 2>&1 || true' EXIT
  until docker exec airgap-bundler ollama list >/dev/null 2>&1; do sleep 1; done
  for m in "${MODELS[@]}"; do
    log "ollama pull $m"
    docker exec airgap-bundler ollama pull "$m"
  done
  docker exec airgap-bundler ollama list | tee "$OUT/MODELS.txt"
  log "Packing models -> $OUT/ollama-models.tar"
  docker exec airgap-bundler tar -C /root/.ollama -cf - models > "$OUT/ollama-models.tar"
  docker rm -f airgap-bundler >/dev/null
  trap - EXIT
fi

# 2. Container images for arm64.
if [[ $(uname -m) != aarch64 && $(uname -m) != arm64 ]]; then
  log "Host is $(uname -m): building arm64 rag-api under emulation"
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null
fi
for img in "${IMAGES[@]:0:3}"; do
  log "docker pull --platform $PLATFORM $img"
  docker pull --platform "$PLATFORM" "$img"
done
log "Building rag-api for $PLATFORM"
docker buildx build --platform "$PLATFORM" --load \
  --build-arg CHROMA_VERSION="$CHROMA_VERSION" -t airgap/rag-api:local rag-api

for img in "${IMAGES[@]}"; do
  arch=$(docker image inspect -f '{{.Architecture}}' "$img")
  [[ $arch == arm64 ]] || { echo "ERROR: $img is $arch, expected arm64" >&2; exit 1; }
done

log "Saving images -> $OUT/images.tar"
# --platform (Docker 28+) keeps the containerd image store from exporting other arches.
docker save --platform "$PLATFORM" -o "$OUT/images.tar" "${IMAGES[@]}" 2>/dev/null \
  || docker save -o "$OUT/images.tar" "${IMAGES[@]}"

# 3. The stack itself (compose file, scripts, rag-api source, configs).
log "Packing stack config -> $OUT/stack.tar.gz"
tar --exclude=./bundle --exclude=./data --exclude=./.env -czf "$OUT/stack.tar.gz" .

{
  echo "Air-gapped AI stack bundle — created $(date -u +%FT%TZ) on $(uname -srm)"
  echo
  echo "Images (linux/arm64):"
  for img in "${IMAGES[@]}"; do
    printf '  %-55s %s\n' "$img" "$(docker image inspect -f '{{.Id}}' "$img")"
  done
} > "$OUT/MANIFEST.txt"
[[ -f $OUT/MODELS.txt ]] && { echo; echo "Models:"; sed 's/^/  /' "$OUT/MODELS.txt"; } >> "$OUT/MANIFEST.txt"

log "Checksums"
(cd "$OUT" && sha256sum ./*.tar ./*.tar.gz MANIFEST.txt > SHA256SUMS)
cat "$OUT/MANIFEST.txt"
du -sh "$OUT"
echo
echo "Done. Copy the '$OUT' folder to removable media, then on the Spark:"
echo "  mkdir airgap-ai && tar -xzf bundle/stack.tar.gz -C airgap-ai"
echo "  cd airgap-ai && ./scripts/install.sh /path/to/bundle"
