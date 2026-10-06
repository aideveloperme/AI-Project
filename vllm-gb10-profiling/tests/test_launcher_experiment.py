import json
from pathlib import Path

from servebench.experiment import deep_merge, load_experiment
from servebench.launcher import build_command, render_flags

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def test_render_flags():
    assert render_flags({"a": True, "b": False, "c": 3, "d": None, "e": {"x": 1}}) == [
        "--a",
        "--no-b",
        "--c",
        "3",
        "--e",
        '{"x":1}',
    ]


def test_deep_merge_keeps_base():
    assert deep_merge({"s": {"a": 1, "b": 2}}, {"s": {"b": 3}}) == {"s": {"a": 1, "b": 3}}


def test_all_experiments_load_and_differ_from_baseline_only_in_declared_knobs():
    base = load_experiment(CONFIGS / "00_baseline.yaml")
    for p in sorted(CONFIGS.glob("0*.yaml")):
        exp = load_experiment(p)
        assert exp["served_model_name"] == base["served_model_name"]
        assert exp["workloads"] == base["workloads"], "workloads must be identical for a fair comparison"
        changed = {k for k in exp["server"] if exp["server"][k] != base["server"].get(k)}
        declared = " ".join(exp["changes"])
        for knob in changed:
            assert knob in declared, f"{p.name}: {knob} changed but not declared in `changes`"
        if exp["model"] != base["model"]:
            assert exp["model"] in declared
        if not p.name.startswith("05_"):  # single-knob studies: one or two things at a time
            assert len(changed) + (exp["model"] != base["model"]) <= 2, (p.name, changed)


def test_docker_command_mounts_profiles_and_uses_vllm_entrypoint(tmp_path):
    exp = load_experiment(CONFIGS / "01_prefix_caching.yaml")
    cmd = build_command(exp, "docker", 8000, tmp_path)
    assert cmd[:2] == ["docker", "run"]
    i = cmd.index("--entrypoint")
    assert cmd[i + 1] == "vllm" and cmd[i + 3 : i + 5] == ["serve", "Qwen/Qwen3-8B"]
    assert "--enable-prefix-caching" in cmd
    pc = json.loads(cmd[cmd.index("--profiler-config") + 1])
    assert pc["profiler"] == "torch" and pc["torch_profiler_dir"] == "/profiles"
    assert f"{tmp_path.resolve()}:/profiles" in cmd


def test_baseline_disables_prefix_cache():
    cmd = build_command(load_experiment(CONFIGS / "00_baseline.yaml"), "local", 8000, None)
    assert "--no-enable-prefix-caching" in cmd and "--profiler-config" not in cmd
