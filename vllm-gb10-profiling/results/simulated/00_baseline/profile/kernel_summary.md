# Kernel summary

## `simulated_rank0.1791288174.pt.trace.json.gz`

- GPU kernel time: **24088.027 ms** over a 25120.027 ms span (busy fraction 0.959)

| category | % of GPU time |
|---|---:|
| gemm | 88.69 |
| attention | 6.49 |
| norm_act | 2.89 |
| sampling | 1.93 |

| kernel | category | calls | total ms | % |
|---|---|---:|---:|---:|
| `sim_gemm_bf16` | gemm | 259 | 21362.8 | 88.69 |
| `sim_flash_attn_paged_kv` | attention | 259 | 1564.206 | 6.49 |
| `sim_rms_norm_kernel` | norm_act | 259 | 696.613 | 2.89 |
| `sim_topk_sampler` | sampling | 259 | 464.409 | 1.93 |
