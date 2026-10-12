# Air-Gapped AI Stack for NVIDIA DGX Spark

A self-contained LLM + RAG stack that runs on a DGX Spark without internet access.
You prepare a bundle on a connected machine, carry it across the air gap on
removable media, and install it on the Spark. After that, nothing pulls from or
reports to the internet.

```
 ┌──────────────── DGX Spark (no internet) ───────────────────────────────┐
 │                                                                         │
 │   browser ──► Open WebUI :3000 ──┐                                      │
 │                                  │        "backend" network             │
 │   curl / apps ─► rag-api :8000 ──┼──►  Ollama (GB10 GPU)  ─ models      │
 │                                  └──►  ChromaDB           ─ vectors     │
 │                                         (internal: no route out)        │
 └─────────────────────────────────────────────────────────────────────────┘
```

| Component | Role | Image |
|---|---|---|
| **Ollama** | Runs LLMs and the embedding model on the Spark's GPU | `ollama/ollama` |
| **Open WebUI** | ChatGPT-style UI, users/auth, document upload & chat-with-docs | `ghcr.io/open-webui/open-webui` |
| **ChromaDB** | Vector store for the RAG API | `chromadb/chroma` |
| **rag-api** | Small FastAPI service (in `rag-api/`): ingest folders/PDFs, ask questions with citations | built locally |

All images are `linux/arm64` (the Spark's Grace CPU is Arm). Versions are pinned in `.env.example`.

## How the air gap is enforced

1. **Physically**: the Spark is never connected to the internet. Everything it
   needs (container images, model weights, Python packages) is in the bundle.
2. **Docker networks**: Ollama and ChromaDB sit only on an `internal` Docker
   network with no route out of the host, even if someone plugs in a cable.
   Only Open WebUI and rag-api join the second network, which is needed to
   publish their ports.
3. **App settings**: Open WebUI runs with `OFFLINE_MODE`, `HF_HUB_OFFLINE`, web
   search / OpenAI connections / update checks / community sharing / telemetry
   all off, and its document RAG uses the local Ollama embedding model instead of
   downloading one from Hugging Face. Chroma telemetry is off.
4. **Integrity**: the bundle carries `SHA256SUMS` and a `MANIFEST.txt` with image
   IDs; `install.sh` refuses to install if any checksum fails.
5. **Exposure**: ports bind to `127.0.0.1` by default. Set `BIND_ADDR` to the
   Spark's address on your isolated LAN to serve other machines there.

`scripts/verify.sh` checks the isolation for you.

## 1. Build the bundle (connected machine)

Requirements: Docker with Buildx, ~100 GB free disk (depends on the models you pick).
This can be any Linux/macOS machine, or the Spark itself before you disconnect it.
On x86 machines the script registers QEMU so it can build the arm64 rag-api image.

```bash
cd airgapped-ai-stack
cp .env.example .env        # optional: change versions / default model
nano models.txt             # pick models (gpt-oss:120b and llama3.3:70b fit on a Spark)
./scripts/bundle.sh
```

Output in `bundle/`:

```
images.tar            all four arm64 images
ollama-models.tar     model weights from models.txt
stack.tar.gz          this folder (compose file, scripts, rag-api source)
MANIFEST.txt          image IDs and model list
SHA256SUMS
```

Copy the `bundle/` folder to a USB drive / approved transfer media. Follow your
organisation's media-transfer procedure (scanning, two-person rule, etc.).

## 2. Install on the DGX Spark (offline)

DGX OS already ships with Docker, the Compose plugin, the NVIDIA driver and the
NVIDIA Container Toolkit, so nothing else needs installing.

```bash
mkdir ~/airgap-ai && tar -xzf /media/$USER/USB/bundle/stack.tar.gz -C ~/airgap-ai
cd ~/airgap-ai
./scripts/install.sh /media/$USER/USB/bundle
./scripts/verify.sh
```

`install.sh` checks the host (arm64, GPU, NVIDIA runtime), verifies checksums,
loads the images, unpacks the models into `data/ollama`, creates `.env` with a
random `WEBUI_SECRET_KEY`, and runs `docker compose up` with `--pull never --no-build`.

Open **http://127.0.0.1:3000** (or `ssh -L 3000:127.0.0.1:3000 user@spark` from
another machine on the isolated LAN). **The first account you create becomes the
admin** — create it right away, then in *Admin Panel → Settings → General* turn
off new sign-ups or set them to "pending" so new users need approval.

## 3. Use it

**Chat / chat with documents** — in Open WebUI, pick a model, and drag files into
the chat or create a *Knowledge* collection (Workspace → Knowledge) for shared document sets.

**RAG API** — for scripts and other internal apps:

```bash
# bulk-index a folder (txt, md, pdf, csv, logs, code, ...)
docker compose run --rm -v /srv/manuals:/docs:ro rag-api python ingest.py /docs

# or upload files
curl -F files=@runbook.pdf -F files=@notes.md http://127.0.0.1:8000/ingest

# ask
curl -s http://127.0.0.1:8000/query -H 'content-type: application/json' \
  -d '{"question": "What is the inspection interval for pump P-101?"}'

# retrieval only, list and remove documents
curl -s http://127.0.0.1:8000/search -H 'content-type: application/json' -d '{"question":"P-101"}'
curl -s http://127.0.0.1:8000/documents
curl -s -X DELETE http://127.0.0.1:8000/documents/runbook.pdf
```

`/query` returns the answer plus the source chunks it was based on, with scores.
Re-ingesting a file with the same name replaces its old chunks. Tune with
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `CHAT_MODEL` env vars on the `rag-api` service.

**Ollama directly**:

```bash
docker exec -it airgap-ollama ollama run gpt-oss:20b
docker exec airgap-ollama ollama ps      # PROCESSOR column should say 100% GPU
```

## Updating (new models or versions)

On the connected machine, edit `models.txt` and/or the versions in `.env`, run
`./scripts/bundle.sh` again (`SKIP_MODELS=1` for images only), carry it over,
and re-run `./scripts/install.sh <bundle>` on the Spark. Your chats, users and
indexed documents in `data/` are kept. Keep `CHROMA_VERSION` in sync between the
Chroma server image and the rag-api client — the scripts use one variable for both.

## Operations

| Task | Command |
|---|---|
| Status / logs | `docker compose ps`, `docker compose logs -f open-webui` |
| Stop / start | `docker compose down` / `docker compose up -d --pull never` |
| GPU / memory | `nvidia-smi`, `docker exec airgap-ollama ollama ps` |
| Backup | stop the stack, then archive `data/` (users, chats, vectors, models) and `.env` |

## Files

```
docker-compose.yml   the stack
.env.example         versions, ports, models, tuning
models.txt           which Ollama models to bundle
rag-api/             FastAPI RAG service (rag.py = logic, app.py = HTTP, ingest.py = CLI)
scripts/bundle.sh    online: pull/build arm64 images, download models, write bundle/
scripts/install.sh   offline: verify, load, start on the Spark
scripts/verify.sh    offline: health, GPU, and isolation checks
```

## Notes and limits

- `rag-api` has no authentication: keep `BIND_ADDR` on localhost or an isolated
  LAN segment, or put it behind your reverse proxy. Open WebUI has its own logins.
- Open WebUI reads most of its `environment:` settings only on first start; after
  that, change them in the Admin Panel.
- What was tested: the rag-api against a real ChromaDB 1.5.9 server (ingest,
  re-ingest, upload, query, delete) and the compose file with `docker compose config`.
  The full stack still needs a first run on the Spark itself: `verify.sh` covers it.
