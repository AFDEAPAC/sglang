#!/usr/bin/env python3
"""Sweep a16wmix gemm1/gemm2 tiles for Kimi-K3.

Default: TP=8 (E=896, dim=3584, inter=384).
TP+EP: --experts 56 --inter-dim 3072 (local E at EP=16, full inter_dim).

Writes FLYDSL_A16WMIX_TUNED_CSV. BM must be a multiple of 16 (MFMA 16x16);
tile_m=8 is illegal. Stays on a16w4 / software MXFP4 upconvert.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import traceback

import torch

from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.moe_a16wmix_host import flydsl_a16w4_gemm1, flydsl_a16w4_gemm2
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

MODEL_DIM = 3584
INTER_DIM = 384
EXPERTS = 896
TOPK = 16
BETA, LINEAR_BETA = 2.0, 1.5
W_DTYPE = "mxfp4"

# Unique kernels to JIT. Serving today is bm=32, gemm1 tn=128 tk=256 bcm-by-token, gemm2 tn=256 tk=128 xcd1.
GEMM1 = [
    # tile_m, tile_n, tile_k, k_wave, b_nt, xcd
    (16, 64, 128, 4, 0, 0),
    (16, 64, 128, 1, 0, 0),
    (16, 128, 256, 1, 0, 0),
    (16, 128, 256, 1, 2, 0),
    (16, 128, 128, 1, 0, 1),
    (32, 64, 128, 4, 0, 0),
    (32, 128, 256, 1, 0, 0),  # closest to current decode
    (32, 128, 256, 1, 2, 0),
    (32, 128, 128, 1, 0, 1),
    (32, 128, 256, 1, 2, 1),
    # fatter N for EP inter_dim=3072
    (16, 256, 128, 1, 0, 0),
    (32, 256, 128, 1, 0, 1),
    (32, 256, 256, 1, 0, 1),
]
GEMM2 = [
    # tile_m, tile_n, tile_k, b_nt, xcd
    (16, 128, 128, 0, 0),
    (16, 256, 128, 0, 1),
    (16, 256, 256, 0, 1),
    (32, 256, 128, 0, 1),  # current fused_moe
    (32, 256, 128, 2, 1),
    (32, 128, 128, 0, 1),
    (32, 256, 64, 0, 1),
    (32, 128, 64, 0, 0),
    (32, 256, 256, 0, 1),
]

DECODE_TOKENS = (1, 2, 4, 8)
PREFILL_TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def log(msg: str) -> None:
    print(msg, flush=True)


def prepare_weights(device):
    torch.manual_seed(0)
    scale = 0.2
    n_out = 2 * INTER_DIM
    w1 = torch.randn((EXPERTS, n_out, MODEL_DIM), device=device, dtype=torch.float32) * scale
    w2 = torch.randn((EXPERTS, MODEL_DIM, INTER_DIM), device=device, dtype=torch.float32) * (
        scale / INTER_DIM**0.5
    )
    w1_q, w1_scale = per_1x32_f4_quant(w1.reshape(EXPERTS * n_out, MODEL_DIM))
    w2_q, w2_scale = per_1x32_f4_quant(w2.reshape(EXPERTS * MODEL_DIM, INTER_DIM))
    del w1, w2
    gc.collect()
    torch.cuda.empty_cache()
    fp4 = dtypes.fp4x2
    w1_sh = shuffle_weight_a16w4(w1_q.view(fp4).reshape(EXPERTS, n_out, -1), 16, False)
    w2_sh = shuffle_weight_a16w4(w2_q.view(fp4).reshape(EXPERTS, MODEL_DIM, -1), 16, False)
    w1_sc = shuffle_scale_a16w4(w1_scale.view(EXPERTS * n_out, MODEL_DIM // 32), EXPERTS, False)
    w2_sc = fp4_utils.e8m0_shuffle(w2_scale.view(EXPERTS * MODEL_DIM, INTER_DIM // 32))
    del w1_q, w2_q, w1_scale, w2_scale
    gc.collect()
    torch.cuda.empty_cache()
    return w1_sh.view(torch.uint8), w1_sc.view(torch.uint8), w2_sh.view(torch.uint8), w2_sc.view(torch.uint8)


def sort_for(tokens, tile_m, hidden, device):
    score = torch.rand((tokens, EXPERTS), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=TOPK, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids.to(torch.int32),
        topk_weights.to(torch.float32),
        EXPERTS,
        MODEL_DIM,
        hidden.dtype,
        tile_m,
        expert_mask=None,
    )
    if num_valid_ids.numel() > 1:
        num_valid_ids = num_valid_ids[:1].contiguous()
    return {
        "hidden": hidden,
        "sorted_ids": sorted_ids.to(torch.int32).contiguous(),
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid": num_valid_ids.to(torch.int32).contiguous(),
        "moe_buf": moe_buf,
        "sorted_size": int(sorted_ids.numel()),
    }


def time_fn(fn, warmup, iters):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def run_gemm1(pack, w1, w1s, tokens, cfg):
    tm, tn, tk, kw, bnt, xcd = cfg
    inter = torch.empty(pack["sorted_size"], INTER_DIM, dtype=torch.bfloat16, device=pack["hidden"].device)
    flydsl_a16w4_gemm1(
        a_bf16=pack["hidden"],
        w1_u8=w1,
        w1_scale_u8=w1s,
        sorted_expert_ids=pack["sorted_expert_ids"],
        cumsum_tensor=pack["num_valid"],
        m_indices=pack["sorted_ids"],
        inter_sorted_bf16=inter,
        n_tokens=tokens,
        NE=EXPERTS,
        D_HIDDEN=MODEL_DIM,
        D_INTER=INTER_DIM,
        topk=TOPK,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        k_wave=kw,
        b_nt=bnt,
        xcd_swizzle=xcd,
        act="situv2",
        situ_beta=BETA,
        situ_linear_beta=LINEAR_BETA,
        w_dtype=W_DTYPE,
        w_layout="standard",
        use_csv_config=False,
    )
    return inter


def run_gemm2(pack, inter, w2, w2s, tokens, cfg):
    tm, tn, tk, bnt, xcd = cfg
    flat = pack["moe_buf"].view(-1)
    flat.zero_()
    flydsl_a16w4_gemm2(
        inter_sorted_bf16=inter,
        w2_u8=w2,
        w2_scale_u8=w2s,
        sorted_expert_ids=pack["sorted_expert_ids"],
        cumsum_tensor=pack["num_valid"],
        sorted_token_ids=pack["sorted_ids"],
        sorted_weights=pack["sorted_weights"],
        flat_out=flat,
        M_logical=tokens,
        max_sorted=pack["sorted_size"],
        NE=EXPERTS,
        D_HIDDEN=MODEL_DIM,
        D_INTER=INTER_DIM,
        topk=TOPK,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        b_nt=bnt,
        xcd_swizzle=xcd,
        w_dtype=W_DTYPE,
        use_csv_config=False,
    )


def try_time(label, fn, warmup, iters):
    try:
        us = time_fn(fn, warmup, iters)
        log(f"  OK {label}  {us:.1f} us")
        return us, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log(f"  FAIL {label}  {err}")
        return None, err


def main():
    global MODEL_DIM, INTER_DIM, EXPERTS, TOPK
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/opt/scripts/configs/a16wmix_tuned.csv")
    ap.add_argument("--log", default="/opt/scripts/logs/tune_a16wmix_moe.log")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--model-dim", type=int, default=MODEL_DIM)
    ap.add_argument("--inter-dim", type=int, default=INTER_DIM)
    ap.add_argument("--experts", type=int, default=EXPERTS)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument(
        "--tokens",
        default="",
        help="comma-separated token counts; default decode+prefill (or decode-only)",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="keep existing CSV rows for other (inter_dim, experts) keys",
    )
    args = ap.parse_args()
    MODEL_DIM = args.model_dim
    INTER_DIM = args.inter_dim
    EXPERTS = args.experts
    TOPK = args.topk
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    os.environ["A16WMIX_FORCE_TILES"] = "1"

    device = torch.device("cuda:0")
    if args.tokens.strip():
        tokens_list = [int(x) for x in args.tokens.split(",") if x.strip()]
    else:
        tokens_list = list(DECODE_TOKENS) if args.decode_only else list(DECODE_TOKENS) + list(PREFILL_TOKENS)
    log(f"gfx tuner a16wmix tokens={tokens_list} warmup={args.warmup} iters={args.iters}")
    log(f"preparing MXFP4 weights E={EXPERTS} model={MODEL_DIM} inter={INTER_DIM} topk={TOPK} ...")
    w1, w1s, w2, w2s = prepare_weights(device)
    log("weights ready")

    rows = []
    detail = []
    packs = {}

    for tokens in tokens_list:
        hidden = torch.randn((tokens, MODEL_DIM), device=device, dtype=torch.bfloat16)
        best1, best2 = None, None
        for tm in sorted({c[0] for c in GEMM1}):
            pack = sort_for(tokens, tm, hidden, device)
            packs[(tokens, tm)] = pack
            # one reference gemm1 output for gemm2 sweeps at this tile_m
            ref_cfg = next((c for c in GEMM1 if c[0] == tm), None)
            inter_ref = None
            if ref_cfg:
                try:
                    inter_ref = run_gemm1(pack, w1, w1s, tokens, ref_cfg)
                    torch.cuda.synchronize()
                except Exception as e:
                    log(f"  ref gemm1 fail tm={tm} tok={tokens}: {type(e).__name__}: {e}")

            for cfg in GEMM1:
                if cfg[0] != tm:
                    continue
                label = f"g1 tok={tokens} tm={cfg[0]} tn={cfg[1]} tk={cfg[2]} kw={cfg[3]} bnt={cfg[4]} xcd={cfg[5]}"
                us, err = try_time(label, lambda c=cfg: run_gemm1(pack, w1, w1s, tokens, c), args.warmup, args.iters)
                detail.append(("gemm1", tokens, *cfg, us, err))
                if us is not None and (best1 is None or us < best1[0]):
                    best1 = (us, cfg)

            if inter_ref is None:
                continue
            for cfg in GEMM2:
                if cfg[0] != tm:
                    continue
                label = f"g2 tok={tokens} tm={cfg[0]} tn={cfg[1]} tk={cfg[2]} bnt={cfg[3]} xcd={cfg[4]}"
                us, err = try_time(
                    label,
                    lambda c=cfg: run_gemm2(pack, inter_ref, w2, w2s, tokens, c),
                    args.warmup,
                    args.iters,
                )
                detail.append(("gemm2", tokens, cfg[0], cfg[1], cfg[2], 1, cfg[3], cfg[4], us, err))
                if us is not None and (best2 is None or us < best2[0]):
                    best2 = (us, cfg)

        log(f"BEST tok={tokens} gemm1={best1} gemm2={best2}")
        if best1:
            us, cfg = best1
            rows.append(
                {
                    "w_dtype": W_DTYPE,
                    "model_dim": MODEL_DIM,
                    "inter_dim": INTER_DIM,
                    "experts": EXPERTS,
                    "topk": TOPK,
                    "token": tokens,
                    "stage": 1,
                    "tile_m": cfg[0],
                    "tile_n": cfg[1],
                    "tile_k": cfg[2],
                    "k_wave": cfg[3],
                    "xcd_swizzle": cfg[5],
                    "b_nt": cfg[4],
                    "us": f"{us:.3f}",
                }
            )
        if best2:
            us, cfg = best2
            rows.append(
                {
                    "w_dtype": W_DTYPE,
                    "model_dim": MODEL_DIM,
                    "inter_dim": INTER_DIM,
                    "experts": EXPERTS,
                    "topk": TOPK,
                    "token": tokens,
                    "stage": 2,
                    "tile_m": cfg[0],
                    "tile_n": cfg[1],
                    "tile_k": cfg[2],
                    "k_wave": 1,
                    "xcd_swizzle": cfg[4],
                    "b_nt": cfg[3],
                    "us": f"{us:.3f}",
                }
            )

        del hidden
        gc.collect()
        torch.cuda.empty_cache()

    fields = [
        "w_dtype",
        "model_dim",
        "inter_dim",
        "experts",
        "topk",
        "token",
        "stage",
        "tile_m",
        "tile_n",
        "tile_k",
        "k_wave",
        "xcd_swizzle",
        "b_nt",
        "us",
    ]
    if args.append and os.path.exists(args.out):
        kept = []
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                same = (
                    int(r["model_dim"]) == MODEL_DIM
                    and int(r["inter_dim"]) == INTER_DIM
                    and int(r["experts"]) == EXPERTS
                    and int(r["topk"]) == TOPK
                    and r.get("w_dtype", W_DTYPE) == W_DTYPE
                )
                if not same:
                    kept.append(r)
        log(f"append: kept {len(kept)} existing rows, add {len(rows)} new")
        rows = kept + rows
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {args.out} rows={len(rows)}")
    det_path = args.out.replace(".csv", "_detail.csv")
    with open(det_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "token", "tile_m", "tile_n", "tile_k", "k_wave", "b_nt", "xcd", "us", "err"])
        for d in detail:
            if d[0] == "gemm1":
                w.writerow(d)
            else:
                w.writerow(d)
    log(f"wrote {det_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
