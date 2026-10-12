# Kernel summary

## `simulated_rank0.1791287918.pt.trace.json.gz`

- GPU kernel time: **24088.412 ms** over a 25120.412 ms span (busy fraction 0.959)

| category | % of GPU time |
|---|---:|
| gemm | 88.7 |
| attention | 6.48 |
| norm_act | 2.89 |
| sampling | 1.93 |

| kernel | category | calls | total ms | % |
|---|---|---:|---:|---:|
| `sim_gemm_bf16` | gemm | 259 | 21366.169 | 88.7 |
| `sim_flash_attn_paged_kv` | attention | 259 | 1561.038 | 6.48 |
| `sim_rms_norm_kernel` | norm_act | 259 | 696.723 | 2.89 |
| `sim_topk_sampler` | sampling | 259 | 464.482 | 1.93 |
