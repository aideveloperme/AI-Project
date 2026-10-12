# Kernel summary

## `simulated_rank0.1791288222.pt.trace.json.gz`

- GPU kernel time: **24169.032 ms** over a 25197.032 ms span (busy fraction 0.959)

| category | % of GPU time |
|---|---:|
| gemm | 88.77 |
| attention | 6.41 |
| norm_act | 2.89 |
| sampling | 1.93 |

| kernel | category | calls | total ms | % |
|---|---|---:|---:|---:|
| `sim_gemm_bf16` | gemm | 258 | 21454.305 | 88.77 |
| `sim_flash_attn_paged_kv` | attention | 258 | 1548.732 | 6.41 |
| `sim_rms_norm_kernel` | norm_act | 258 | 699.597 | 2.89 |
| `sim_topk_sampler` | sampling | 258 | 466.398 | 1.93 |
