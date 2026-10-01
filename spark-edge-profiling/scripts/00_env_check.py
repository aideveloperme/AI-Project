#!/usr/bin/env python3
"""Step 0 — inventory the DGX Spark software/hardware stack and probe every tool we need."""

import shutil

import _bootstrap  # noqa: F401
from sparkprof.config import base_parser, config_from_args, save_json
from sparkprof.hardware import device_info
from sparkprof.power import PowerSampler, idle_power


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    info = device_info()
    info["tools"] = {t: shutil.which(t) for t in ("nvidia-smi", "ncu", "nsys", "trtexec")}
    ps = PowerSampler()
    info["power_backend"] = ps.backend
    info["idle_power_w"] = idle_power(2.0) if ps.available else None

    required = {"cuda_available": info["cuda_available"], "tensorrt": info.get("tensorrt"),
                "modelopt": info.get("modelopt"), "triton": info.get("triton"),
                "ncu": info["tools"]["ncu"], "power": ps.backend}
    print("\n=== sparkprof environment ===")
    for k, v in info.items():
        if k != "tools":
            print(f"  {k:18s} {v}")
    print("\n=== readiness ===")
    for k, v in required.items():
        print(f"  [{'ok' if v else 'MISSING':7s}] {k}")
    if not info["cuda_available"]:
        print("\nNo CUDA device — run inside the NGC container on the DGX Spark (see docs/SETUP_DGX_SPARK.md)")
    save_json(info, f"{cfg.paths.results_dir}/env.json")


if __name__ == "__main__":
    main()
