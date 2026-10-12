"""End-to-end: harness <-> simulator over real HTTP/SSE."""

from pathlib import Path

import yaml

from servebench.agent import AgentConfig, run_agent
from servebench.experiment import load_experiment, run_experiment_async
from servebench.loadgen import run_level
from servebench.report import build_report
from servebench.workloads import WorkloadSpec

RAG = WorkloadSpec(
    name="rag", kind="shared_prefix", shared_prefix_tokens=1024, num_prefixes=1, input_tokens=32, output_tokens=16
)


async def test_loadtest_and_prefix_cache_effect(mock_server_factory):
    off = mock_server_factory("--no-enable-prefix-caching")
    on = mock_server_factory("--enable-prefix-caching")
    r_off = await run_level(off, "qwen3-8b", RAG, level=4, num_requests=8)
    r_on = await run_level(on, "qwen3-8b", RAG, level=4, num_requests=8)
    assert r_off.summary["num_ok"] == 8 and r_on.summary["num_ok"] == 8
    assert r_on.summary["mean_output_tokens"] == 16  # ignore_eos honoured
    assert r_off.server_counters["prefix_cache_hit_rate"] == 0
    assert r_on.server_counters["prefix_cache_hit_rate"] > 0.8
    assert r_on.summary["cached_prompt_token_frac"] > 0.8


async def test_context_limit_rejects_with_400(mock_server_factory):
    url = mock_server_factory("--max-model-len", "512")
    spec = WorkloadSpec(name="long", kind="long_context", input_tokens=2000, output_tokens=8)
    r = await run_level(url, "qwen3-8b", spec, level=2, num_requests=2, warmup_requests=0)
    assert r.summary["num_ok"] == 0
    assert all(x.status_code == 400 for x in r.results)


async def test_agent_modes(mock_server_factory):
    url = mock_server_factory("--enable-prefix-caching")
    replay = await run_agent(
        url,
        "qwen3-8b",
        AgentConfig(mode="replay", sessions=2, episodes_per_session=1, replay_steps=3, system_prompt_tokens=500),
    )
    s = replay["summary"]
    assert s["episodes_ok"] == 2 and [p["step"] for p in s["per_step"]] == [1, 2, 3]
    assert s["per_step"][2]["mean_prompt_tokens"] > s["per_step"][0]["mean_prompt_tokens"]
    assert replay["server"]["prefix_cache_hit_rate"] > 0.5
    react = await run_agent(
        url, "qwen3-8b", AgentConfig(mode="react", sessions=1, episodes_per_session=1, system_prompt_tokens=200)
    )
    assert react["summary"]["episodes_ok"] == 1
    assert any(st["tool_calls"] for st in react["steps"])  # tool-call deltas were parsed and executed


async def test_full_experiment_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVEBENCH_MOCK_TIME_SCALE", "0.02")
    exp = load_experiment(Path(__file__).resolve().parents[1] / "configs/experiments/01_prefix_caching.yaml")
    exp["workloads"] = [{**RAG.__dict__, "sweep": {"mode": "concurrency", "levels": [2], "requests_per_level": 4}}]
    exp["agent"] = [
        {"mode": "replay", "sessions": 1, "episodes_per_session": 1, "replay_steps": 2, "system_prompt_tokens": 200}
    ]
    exp["profile"].update({"workload": "rag", "requests": 4, "concurrency": 2})
    s = await run_experiment_async(
        exp, tmp_path / "01_prefix_caching", launcher_name="mock", port=18765, log=lambda *_: None
    )
    assert s["simulated"] is True and s["loadtest"][0]["num_ok"] == 4
    assert s["profile"]["traces"], "simulated profiler trace should be captured"
    assert 0 < s["profile"]["summaries"][0]["gpu_busy_frac"] <= 1
    assert (tmp_path / "01_prefix_caching/profile/kernel_summary.md").exists()
    assert yaml.safe_load((tmp_path / "01_prefix_caching/experiment.resolved.yaml").read_text())["name"]
    report = build_report(tmp_path).read_text()
    assert "SIMULATED RESULTS" in report and "rag" in report
