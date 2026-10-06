"""Start/stop a vLLM server for one experiment.

Launchers
---------
docker    ``docker run ... --entrypoint vllm <image> serve <model> <flags>`` (recommended on DGX Spark)
local     ``vllm serve <model> <flags>`` from the current Python env
mock      the GB10 simulator in ``servebench.mock`` (CI / harness development, no GPU)
external  do nothing; benchmark a server someone else started
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

CONTAINER_NAME = "servebench-vllm"
CONTAINER_PROFILE_DIR = "/profiles"


def render_flags(server: dict[str, Any]) -> list[str]:
    """``{"max-model-len": 8192, "enable-prefix-caching": False}`` -> CLI flags.

    True -> ``--flag``; False -> ``--no-flag`` (vLLM's BooleanOptionalAction);
    None -> omitted; dict/list -> JSON.
    """
    out: list[str] = []
    for key, val in server.items():
        flag = f"--{key}"
        if val is None:
            continue
        if val is True:
            out.append(flag)
        elif val is False:
            out.append(f"--no-{key}")
        elif isinstance(val, (dict, list)):
            out += [flag, json.dumps(val, separators=(",", ":"))]
        else:
            out += [flag, str(val)]
    return out


@dataclass
class ServerHandle:
    launcher: str
    cmd: list[str]
    proc: subprocess.Popen | None
    log: IO[str] | None

    def alive(self) -> bool:
        return self.proc is None or self.proc.poll() is None

    def stop(self, timeout_s: float = 90.0) -> None:
        if self.launcher == "docker":
            subprocess.run(
                ["docker", "stop", "-t", str(int(timeout_s)), CONTAINER_NAME], capture_output=True, check=False
            )
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
                self.proc.wait(timeout=timeout_s)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        if self.log:
            self.log.close()


def build_command(exp: dict[str, Any], launcher: str, port: int, profile_dir: Path | None) -> list[str]:
    server = dict(exp.get("server", {}))
    if profile_dir is not None:
        pdir = CONTAINER_PROFILE_DIR if launcher == "docker" else str(profile_dir.resolve())
        server["profiler-config"] = {
            **exp.get("profile", {}).get("profiler_config", {}),
            "profiler": "torch",
            "torch_profiler_dir": pdir,
        }
    flags = [
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--served-model-name",
        exp["served_model_name"],
        *render_flags(server),
    ]
    model = exp["model"]
    if launcher == "local":
        return ["vllm", "serve", model, *flags]
    if launcher == "mock":
        return [sys.executable, "-m", "servebench.mock", model, *flags, *render_flags(exp.get("mock", {}))]
    if launcher == "docker":
        d = exp.get("docker", {})
        image = os.environ.get("VLLM_IMAGE", d.get("image", "vllm/vllm-openai:v0.31.0"))
        hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            CONTAINER_NAME,
            "--gpus",
            "all",
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "-p",
            f"{port}:{port}",
            "-v",
            f"{hf_home}:/root/.cache/huggingface",
        ]
        if os.environ.get("HF_TOKEN"):
            cmd += ["-e", "HF_TOKEN"]
        for k, v in (d.get("env") or {}).items():
            cmd += ["-e", f"{k}={v}"]
        if profile_dir is not None:
            cmd += ["-v", f"{profile_dir.resolve()}:{CONTAINER_PROFILE_DIR}"]
        cmd += [*d.get("extra_args", []), "--entrypoint", "vllm", image, "serve", model, *flags]
        return cmd
    raise ValueError(f"unknown launcher {launcher!r}")


def launch(
    exp: dict[str, Any], launcher: str, port: int, log_path: Path, profile_dir: Path | None = None
) -> ServerHandle:
    if launcher == "external":
        return ServerHandle("external", [], None, None)
    cmd = build_command(exp, launcher, port, profile_dir)
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
    if launcher == "docker":  # clean up a container left behind by a crashed run
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    log.write("$ " + shlex.join(cmd) + "\n\n")
    log.flush()
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (exp.get("env") or {}).items()})
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    time.sleep(0.5)
    return ServerHandle(launcher, cmd, proc, log)
