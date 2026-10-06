import gzip
import json

from servebench.profiling import categorize, render_markdown, summarize_trace


def test_categorize():
    assert categorize("flash_fwd_splitkv_kernel") == "attention"
    assert categorize("sm90_xmma_gemm_bf16bf16") == "gemm"
    assert categorize("ncclDevKernel_AllReduce") == "communication"
    assert categorize("vllm::rms_norm_kernel") == "norm_act"
    assert categorize("weird_thing") == "other"


def test_summarize_trace(tmp_path):
    ev = [
        {"ph": "X", "cat": "kernel", "name": "gemm_a", "ts": 0, "dur": 600},
        {"ph": "X", "cat": "kernel", "name": "flash_attn", "ts": 600, "dur": 300},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 5000},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 950, "dur": 50},
    ]
    p = tmp_path / "r0.pt.trace.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump({"traceEvents": ev}, f)
    s = summarize_trace(p)
    assert s["gpu_kernel_time_ms"] == 0.95
    assert s["gpu_busy_frac"] == 0.95
    assert s["by_category_pct"]["gemm"] == round(100 * 600 / 950, 2)
    assert "gemm_a" in render_markdown([s])
