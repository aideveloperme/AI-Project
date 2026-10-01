"""GPU-saturating training workload for GPU Sentinel (DGX Spark / any NVIDIA GPU).

Trains a GPT-style decoder on synthetic tokens forever (or for --minutes), in
bf16, with data generated on the GPU so nothing on the host can slow it down.
It keeps the GPU at ~100% utilization and exposes Prometheus metrics that GPU
Sentinel understands, so the dashboard shows real throughput and step time:

    sentinel_workload_throughput      tokens/s
    sentinel_workload_step_time_ms    ms per optimizer step
    sentinel_dataloader_wait_ratio    fraction of the step spent waiting for data
    sentinel_workload_info / sentinel_node_info   job + node inventory

Run (inside an NVIDIA PyTorch container):
    python train_gpt.py                       # defaults sized for a DGX Spark
    python train_gpt.py --size large --batch 16
    python train_gpt.py --minutes 30

Metrics: http://<host>:9500/metrics   (no extra Python packages needed)
"""
from __future__ import annotations

import argparse
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {  # layers, d_model, heads
    "tiny": (2, 128, 4),  # smoke tests only
    "small": (8, 768, 12),
    "medium": (16, 1024, 16),
    "large": (24, 1536, 16),
}


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab: int, seq: int, layers: int, d: int, heads: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        for m in self.modules():  # GPT-2 style init
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


# --------------------------------------------------------------- metrics
class Metrics:
    def __init__(self, labels: dict[str, str]):
        self.labels = labels
        self.values: dict[str, float] = {}
        self.lock = threading.Lock()

    def set(self, **kv: float) -> None:
        with self.lock:
            self.values.update(kv)

    def render(self) -> str:
        def lab(d: dict[str, str]) -> str:
            return "{" + ",".join(f'{k}="{v}"' for k, v in d.items()) + "}"
        L = self.labels
        wl = {"job_id": L["job_id"]}
        with self.lock:
            v = dict(self.values)
        out = [
            "# TYPE sentinel_workload_info gauge",
            f'sentinel_workload_info{lab({"job_id": L["job_id"], "name": L["name"], "scheduler": "standalone", "framework": "pytorch", "kind": "training", "nodes": L["node"], "throughput_unit": "tokens/s"})} 1',
            "# TYPE sentinel_node_info gauge",
            f'sentinel_node_info{lab({"cluster": L["cluster"], "server_type": L["server_type"], "gpu_model": L["gpu_model"], "job_id": L["job_id"]})} 1',
        ]
        for name, key in (("sentinel_workload_throughput", "tokens_per_s"), ("sentinel_workload_step_time_ms", "step_ms"),
                          ("sentinel_dataloader_wait_ratio", "wait_ratio"), ("sentinel_workload_loss", "loss"),
                          ("sentinel_workload_steps_total", "steps")):
            if key in v:
                out += [f"# TYPE {name} gauge", f"{name}{lab(wl)} {v[key]:.6g}"]
        return "\n".join(out) + "\n"


def serve(metrics: Metrics, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = metrics.render().encode() if self.path.startswith("/metrics") else b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=SIZES, default="medium")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--minutes", type=float, default=0, help="stop after N minutes (0 = run forever)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (slower start, faster steps)")
    ap.add_argument("--metrics-port", type=int, default=int(os.getenv("METRICS_PORT", "9500")))
    ap.add_argument("--job-id", default=os.getenv("SENTINEL_JOB_ID", "spark-train-1"))
    ap.add_argument("--node", default=os.getenv("SENTINEL_NODE", "spark-01"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    layers, d, heads = SIZES[args.size]
    model = GPT(args.vocab, args.seq, layers, d, heads).to(dev)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1,
                            fused=dev.type == "cuda")
    step_fn = torch.compile(model) if args.compile else model

    metrics = Metrics({"job_id": args.job_id, "name": f"gpt-{args.size}-synthetic", "node": args.node,
                       "cluster": "dgx-spark", "server_type": "DGX Spark", "gpu_model": gpu_name})
    threading.Thread(target=serve, args=(metrics, args.metrics_port), daemon=True).start()
    print(f"[train] {gpu_name} · GPT-{args.size} {params:.0f}M params · batch {args.batch} × seq {args.seq} · "
          f"metrics on :{args.metrics_port}/metrics", flush=True)

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, args.vocab, (args.batch, args.seq + 1), device=dev)
        return x[:, :-1], x[:, 1:]

    deadline = time.time() + args.minutes * 60 if args.minutes else math.inf
    step, window_t0, window_tokens, window_wait = 0, time.time(), 0, 0.0
    while time.time() < deadline:
        t0 = time.perf_counter()
        x, y = batch()
        t_data = time.perf_counter() - t0
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = step_fn(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        window_tokens += args.batch * args.seq
        window_wait += t_data
        if step % 10 == 0:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - window_t0
            tps = window_tokens / dt
            metrics.set(tokens_per_s=tps, step_ms=dt / 10 * 1000, wait_ratio=min(1.0, window_wait / dt),
                        loss=float(loss.item()), steps=step)
            if step % 100 == 0:
                print(f"[train] step {step:>6}  loss {loss.item():.3f}  {tps:,.0f} tok/s  {dt / 10 * 1000:.0f} ms/step",
                      flush=True)
            window_t0, window_tokens, window_wait = time.time(), 0, 0.0
    print("[train] done", flush=True)


if __name__ == "__main__":
    main()
