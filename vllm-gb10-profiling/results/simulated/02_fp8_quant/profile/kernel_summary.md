# Kernel summary

## `simulated_rank0.1791287181.pt.trace.json.gz`

- GPU kernel time: **12044.025 ms** over a 13076.025 ms span (busy fraction 0.921)

| category | % of GPU time |
|---|---:|
| gemm | 88.69 |
| attention | 6.49 |
| norm_act | 2.89 |
| sampling | 1.93 |

| kernel | category | calls | total ms | % |
|---|---|---:|---:|---:|
| `sim_scaled_mm_fp8` | gemm | 259 | 10681.408 | 88.69 |
| `sim_flash_attn_paged_kv` | attention | 259 | 782.105 | 6.49 |
| `sim_rms_norm_kernel` | norm_act | 259 | 348.307 | 2.89 |
| `sim_topk_sampler` | sampling | 259 | 232.205 | 1.93 |
