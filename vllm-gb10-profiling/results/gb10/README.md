# GB10 (DGX Spark) results

**Not yet populated.** These are filled by running the suite on the device:

```bash
make gb10-setup        # once: pull image, download Qwen3-8B + Qwen3-8B-FP8, sanity checks
make gb10              # ~2-3 h; or `make gb10-quick` (~20-30 min) first
make nsys              # optional Nsight Systems captures
git add results/gb10 && git commit -m "GB10 results" && git push
python -m servebench links results/gb10 --repo <owner>/<repo> --ref <commit-sha>   # Perfetto links
```

`REPORT.md` and per-experiment directories will appear here. See `../README.md` for the layout.
