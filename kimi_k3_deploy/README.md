# Kimi-K3 serving on 2x MI308X (TP8 x PP2): scripts and env

Snapshot of the production serving setup on 2026-10-12, matching image
`...avo20261012_wavefhijmnrl_a8w8_mlabal_pprecv_pf8` (sha256:d33623ac). The source changes these
scripts switch on are in this branch (sglang) and in the `kimi` branch of AFDEAPAC/aiter.

## Where the files go (inside the container on both nodes)

| File | Location |
|---|---|
| `common.env`, `start_*.sh`, `stop_kimi_k3.sh`, `healthcheck.sh`, `run_*.sh`, `*.py` | `/opt/scripts/` |
| `hipblaslt_tuning.txt` | `/opt/hipblaslt_tuning.txt` (`HIPBLASLT_TUNING_OVERRIDE_FILE`, read by `start_kimi_k3_tp8_pp2.sh`) |
| tuned CSVs: aiter `kimi` branch, `aiter/configs/kimi_k3_gfx942/` | `/opt/scripts/configs/` (defaults in `common.env`), or point `FLYDSL_A16WMIX_TUNED_CSV`, `AITER_CONFIG_GEMM_BF16`, `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` at them |

## Start / stop

- `restore_production_server.sh` runs on the node0 host: it stops any server, then starts
  `start_node0.sh` in container `$C` on node0 and `start_node1.sh` on `$NODE2` with the production
  launch env (`SRV_ENV`: PP partition 47,46, CUDA graph max bs 32, a16wmix fp8, ...), and waits for
  `/health`.
- `start_node0.sh` / `start_node1.sh` call `start_kimi_k3_tp8_pp2.sh <rank>`, which sources `common.env`.
- `stop_kimi_k3.sh` stops the local server.

## Landed optimizations

Each `land_*.py` adds (default), checks (`--check`) or removes (`--revert`) one env block in
`common.env`; the blocks in the shipped `common.env` are the landed state. `set_wave_f_env.py` manages a
separate temporary "arm" block for A/B runs (`--clear` removes it).

## Site-specific values

`common.env`, `start_*.sh` and `restore_production_server.sh` contain this cluster's node IPs
(10.170.168.1 / .2), InfiniBand HCA names, container name and model path. Change them for another cluster.
