# Kimi-K3 serving on two AMD Instinct MI300X nodes (TP8 x PP2): scripts and environment

Launch scripts and defaults for serving Kimi-K3 with SGLang on two MI300X (gfx942) nodes: tensor
parallelism inside each node and two pipeline stages across them. The kernels these scripts switch on are
in this branch and in the branch of the same name of the AITER fork.

## Files

| File | Use |
|---|---|
| `common.env` | Defaults for every setting, sourced by the start script. Override any value from the environment. |
| `start_kimi_k3_tp8_pp2.sh <rank>` | Starts one pipeline stage; `start_node0.sh` and `start_node1.sh` call it with rank 0 and 1. |
| `k3_launch.sh <rank> [dspark\|base]` | Starts one rank with a profile: `base` (radix cache, decode CUDA graphs up to batch 32, prefix attention, DCP8, HiCache) or `dspark` (the same plus DSpark speculative decoding). |
| `profile_base.env`, `profile_dspark.env` | The settings of those two profiles. |
| `stop_kimi_k3.sh` | Stops the SGLang processes on this node. |
| `healthcheck.sh`, `run_bench_serving.sh`, `run_eval_gsm8k.sh` | Health check, a short serving benchmark, and GSM8K against the running server. |
| `tune_a16wmix_moe.py`, `tune_kimik3_bf16_gemm.sh` | Tuning tools for the a16wmix MoE tiles and the dense bf16 GEMMs. |

## Start

Run the same script on both nodes inside the container, from the root of this branch, node 0 first:

```bash
export MODEL_PATH=<model dir>
export MASTER_ADDR=<data IP of node 0> DIST_INIT_ADDR=<data IP of node 0>:<port>
export NODE0_IP=<data IP of node 0> NODE1_IP=<data IP of node 1>
export NETWORK_IFACE=<data interface>
export ENABLE_NCCL_IB=1 NCCL_NET=IB NCCL_IB_HCA=<RDMA devices> NCCL_IB_GID_INDEX=<RoCE v2 GID index>

T=<AITER checkout>/aiter/configs/kimi_k3_gfx942
export FLYDSL_A16WMIX_TUNED_CSV=$T/a16wmix_tuned.csv
export FLYDSL_A16WMIX_FP8_TUNED_CSV=$T/moe_fp8_table_mi300x.csv
export AITER_CONFIG_GEMM_BF16=$T/kimik3_bf16_tuned_gemm_gfx942.csv
export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=$T/a8w8_bpreshuffle_tuned_gfx942.csv

bash kimi_k3_deploy/start_kimi_k3_tp8_pp2.sh 0    # on node 0 (stage 0)
bash kimi_k3_deploy/start_kimi_k3_tp8_pp2.sh 1    # on node 1 (stage 1)
```

The start script stops with an error if `MODEL_PATH`, `MASTER_ADDR`, `NODE0_IP`, `NODE1_IP`,
`NETWORK_IFACE` or, over RDMA, `NCCL_IB_HCA` is not set. Logs and PID files go to `LOG_DIR`
(default `./logs`). Set `ENABLE_NCCL_IB=0` to run the pipeline traffic over TCP.

## Kimi-K3 switches

The `export` lines at the end of `common.env` are the Kimi-K3-specific switches. Each is on by default;
set it to 0 (or to an empty list) to turn that change off. Most of them print an `[k3-...] active` marker
the first time their code path runs, so count the markers in the server log to check that they are in effect:

```bash
grep -ohE '\[k3-[a-z0-9-]+\] active( mode=[0-9]+)?' <server log> | sort | uniq -c
```
