# Kernel summary

## `simulated_rank0.1791287094.pt.trace.json.gz`

- GPU kernel time: **12084.509 ms** over a 13108.509 ms span (busy fraction 0.922)

| category | % of GPU time |
|---|---:|
| gemm | 88.79 |
| attention | 6.39 |
| norm_act | 2.9 |
| sampling | 1.93 |

| kernel | category | calls | total ms | % |
|---|---|---:|---:|---:|
| `sim_scaled_mm_fp8` | gemm | 257 | 10729.487 | 88.79 |
| `sim_flash_attn_paged_kv` | attention | 257 | 771.898 | 6.39 |
| `sim_rms_norm_kernel` | norm_act | 257 | 349.875 | 2.9 |
| `sim_topk_sampler` | sampling | 257 | 233.25 | 1.93 |
