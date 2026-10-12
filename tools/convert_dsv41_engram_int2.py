#!/usr/bin/env python3
"""Offline converter for DeepSeek-V4.1-Flash FP4PTPC -> Engram INT2 (dsv41_engram_int2_v1).

Produces a complete standalone checkpoint where the two Engram embed tables
(layers 1 and 14) are re-quantized from source FP8 (e4m3 payload + e8m0 group32
scale) to the frozen INT2 contract:

  weight : u8 [rows, 256//4=64]   4 consecutive 2-bit codes per byte, LSB-first
  scale  : u8 [rows, 256//32=8]   e8m0 exponent bytes
  codebook [-1, -r, +r, +1], r=0.30, group=32, dequant rounded once to bf16

Every other tensor is preserved byte-for-byte. The index is rebuilt, config.json
gains the INT2 scheme under text_config, ancillary files are copied, and a
conversion manifest records full provenance.

Quantization truth is the single shared implementation
`sglang.kernels.ops.embeddings.engram_int2.quantize_int2`; this tool never
re-implements the math.

Design guarantees:
- resumable: completed valid shards are validated and skipped, never overwritten;
- atomic: every output file is written to a unique .tmp, fsynced, then renamed;
- shard-safe: a valid final shard is never truncated or restarted;
- GPU-chunked: the big embed tables are converted in bounded row chunks;
- byte-exact: unchanged tensors are copied by raw byte range.

All code / logs / messages are English by project policy.
"""

import argparse
import hashlib
import json
import os
import resource
import shutil
import struct
import sys
import time

import torch
from safetensors import safe_open

from sglang.kernels.ops.embeddings.engram_int2 import (
    quantize_int2,
    dequant_int2_torch,
    engram_int2_gather,
    SCHEME,
    INT2_DIM,
    INT2_GROUP,
    DEFAULT_R,
)

PACK = INT2_DIM // 4            # 64
SCALE_COLS = INT2_DIM // INT2_GROUP  # 8
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
MANIFEST_NAME = "engram_int2_conversion_manifest.json"
CONVERT_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")
# Ancillary files/dirs copied verbatim for a standalone checkpoint. The 48 shards,
# the index (rebuilt) and config.json (modified) are handled separately; .cache is
# a Hugging Face upload cache and is intentionally skipped.
ANCILLARY = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
    "DeepSeek_V41_Tech_Report.pdf",
    "ptpc_conversion_manifest.json",
    "assets",
    "encoding",
    "evaluation",
    "inference",
]
HASH_BLOCK = 64 << 20          # 64 MiB streaming hash/copy block
COPY_BLOCK = 64 << 20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)


def free_gib(path):
    return shutil.disk_usage(path).free / (1 << 30)


def gpu_mem_gib(device):
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.cuda.memory_allocated(device) / (1 << 30)
    return 0.0


def read_st_header(path):
    """Return (header_len, header_dict, data_start) for a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return n, hdr, 8 + n


def sha256_file(path, block=HASH_BLOCK):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_range(fh, start, nbytes, block=HASH_BLOCK):
    """SHA256 of nbytes from an open binary file handle starting at `start`."""
    h = hashlib.sha256()
    fh.seek(start)
    left = nbytes
    while left:
        b = fh.read(min(block, left))
        if not b:
            break
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def dtype_nbytes_from_source(info):
    o0, o1 = info["data_offsets"]
    return o1 - o0


def converted_spec(name, src_info):
    """Return output (dtype, shape, nbytes) for a converted tensor."""
    rows = src_info["shape"][0]
    if name.endswith(".engram.embed.weight"):
        return "U8", [rows, PACK], rows * PACK
    if name.endswith(".engram.embed.scale"):
        return "U8", [rows, SCALE_COLS], rows * SCALE_COLS
    raise ValueError(name)


def scan_source(src):
    """Inspect the source checkpoint. Returns a dict describing layout."""
    idx_path = os.path.join(src, INDEX_NAME)
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    shards = sorted(set(wm.values()))
    convert_tensors = {k: v for k, v in wm.items() if k.endswith(CONVERT_SUFFIXES)}
    convert_shards = sorted(set(convert_tensors.values()))
    # Read headers only for the shards that contain converted tensors.
    shard_headers = {}
    for sh in convert_shards:
        _, hdr, _ = read_st_header(os.path.join(src, sh))
        shard_headers[sh] = hdr
    return {
        "index": idx,
        "weight_map": wm,
        "shards": shards,
        "convert_tensors": convert_tensors,
        "convert_shards": convert_shards,
        "convert_shard_headers": shard_headers,
    }


def predict_sizes(src, scan):
    """Predict per-shard and total output bytes without writing anything."""
    wm = scan["weight_map"]
    convert_shards = set(scan["convert_shards"])
    # source tensor bytes per shard = sum of data_offset spans
    src_shard_bytes = {}
    out_shard_bytes = {}
    out_total = 0
    for sh in scan["shards"]:
        _, hdr, _ = read_st_header(os.path.join(src, sh))
        names = [k for k in hdr if k != "__metadata__"]
        sb = 0
        ob = 0
        for name in names:
            info = hdr[name]
            nb = dtype_nbytes_from_source(info)
            sb += nb
            if name.endswith(CONVERT_SUFFIXES):
                _, _, onb = converted_spec(name, info)
                ob += onb
            else:
                ob += nb
        src_shard_bytes[sh] = sb
        out_shard_bytes[sh] = ob
        out_total += ob
    return src_shard_bytes, out_shard_bytes, out_total


def do_dry_run(src, out, scan):
    log("=== DRY RUN: no destination data will be created ===")
    cfg = json.load(open(os.path.join(src, CONFIG_NAME)))
    tc = cfg.get("text_config", {})
    print(f"architecture        : {cfg.get('architectures')}")
    print(f"model_type          : {cfg.get('model_type')}")
    print(f"quantization_config : {cfg.get('quantization_config')}")
    print(f"text_config.hidden_size={tc.get('hidden_size')} head_dim={tc.get('head_dim')} hc_mult={tc.get('hc_mult')}")
    print(f"INT2 scheme         : {SCHEME} dim={INT2_DIM} group={INT2_GROUP} r={DEFAULT_R}")
    print("\n--- Engram tensors to convert ---")
    for sh in scan["convert_shards"]:
        hdr = scan["convert_shard_headers"][sh]
        for name in [k for k in hdr if k != "__metadata__"]:
            if name.endswith(CONVERT_SUFFIXES):
                info = hdr[name]
                od, os_, onb = converted_spec(name, info)
                print(f"  {name}")
                print(f"    source {info['dtype']:8s} {info['shape']}  {dtype_nbytes_from_source(info)} B")
                print(f"    output {od:8s} {os_}  {onb} B")
    src_sb, out_sb, out_total = predict_sizes(src, scan)
    print("\n--- predicted shard sizes (tensor bytes) ---")
    for sh in scan["convert_shards"]:
        print(f"  {sh}: source {src_sb[sh]} B -> output {out_sb[sh]} B")
    unchanged = [s for s in scan["shards"] if s not in set(scan["convert_shards"])]
    unchanged_bytes = sum(out_sb[s] for s in unchanged)
    largest_unchanged = max(out_sb[s] for s in unchanged)
    print(f"\nunchanged shards       : {len(unchanged)} (copied byte-exact), {unchanged_bytes} B")
    print(f"largest unchanged shard: {largest_unchanged} B ({largest_unchanged/(1<<30):.2f} GiB)")
    print(f"predicted output total : {out_total} B ({out_total/(1<<30):.2f} GiB)")
    peak = out_total + largest_unchanged  # final data + one in-flight .tmp copy
    print(f"predicted peak usage   : ~{peak/(1<<30):.2f} GiB (final + one in-flight shard .tmp)")
    fg = free_gib("/SHARE")
    print(f"/SHARE free now        : {fg:.2f} GiB; retained after peak ~{fg - peak/(1<<30):.2f} GiB")
    print("\n--- preserved tensor families (unchanged, byte-exact) ---")
    fams = {}
    for name in scan["weight_map"]:
        if name.endswith(CONVERT_SUFFIXES):
            continue
        key = name.split(".")[-1]
        fams[key] = fams.get(key, 0) + 1
    for k, c in sorted(fams.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  *.{k}: {c}")
    print("\n--- output config/index changes ---")
    print("  config.json text_config += {engram_quant_scheme, engram_quant_group_size=32, engram_quant_codebook_r=0.3}")
    print(f"  index metadata.total_size: {scan['index']['metadata']['total_size']} -> {out_total}")
    print("  index weight_map: same tensor->shard mapping, converted tensors keep names")
    print("\nDRY RUN OK: layout recognized, no unknown Engram shapes.")
    # sanity: reject unknown layout
    for sh in scan["convert_shards"]:
        hdr = scan["convert_shard_headers"][sh]
        for name in [k for k in hdr if k != "__metadata__"]:
            if name.endswith(".engram.embed.weight"):
                shp = hdr[name]["shape"]
                if len(shp) != 2 or shp[1] != INT2_DIM:
                    raise SystemExit(f"UNKNOWN LAYOUT: {name} shape {shp}, expected [rows,{INT2_DIM}]")
    return out_total


def copy_range(fs, fo, src_start, nbytes, out_start):
    """Copy nbytes from fs@src_start to fo@out_start, returning the SHA256."""
    h = hashlib.sha256()
    fs.seek(src_start)
    fo.seek(out_start)
    left = nbytes
    while left:
        b = fs.read(min(COPY_BLOCK, left))
        if not b:
            raise IOError("short read during range copy")
        fo.write(b)
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def convert_embed_pair(src_path, fo, data0, out_entries, device, chunk_rows, mon):
    """Stream-convert the one Engram embed pair in this shard into fo."""
    wname = [n for n in out_entries if n.endswith(".engram.embed.weight")][0]
    sname = wname[: -len(".weight")] + ".scale"
    we = out_entries[wname]
    se = out_entries[sname]
    rows = we["shape"][0]
    dim = INT2_DIM
    w_off = data0 + we["out_off"]
    s_off = data0 + se["out_off"]
    t0 = time.time()
    done = 0
    with safe_open(src_path, framework="pt") as f:
        wsl = f.get_slice(wname)
        ssl = f.get_slice(sname)
        for st in range(0, rows, chunk_rows):
            c = min(chunk_rows, rows - st)
            w = wsl[st : st + c]                       # cpu fp8_e4m3 [c,256]
            s = ssl[st : st + c]                       # cpu e8m0 [c,8]
            s_u8 = s.view(torch.uint8)                 # reinterpret bytes, cpu
            w = w.to(device, non_blocking=True)
            s_u8 = s_u8.to(device, non_blocking=True)
            wf = w.float()
            scale = torch.exp2(s_u8.to(torch.int32).float() - 127.0)   # [c,8]
            ref = (wf.view(c, dim // INT2_GROUP, INT2_GROUP) * scale.unsqueeze(-1)).reshape(c, dim)
            packed, scale_u8 = quantize_int2(ref)       # [c,64] u8, [c,8] u8
            if not (torch.isfinite(ref).all()):
                raise FloatingPointError(f"non-finite ref at rows {st}:{st+c} of {wname}")
            pb = packed.to("cpu").contiguous().numpy().tobytes()
            sb = scale_u8.to("cpu").contiguous().numpy().tobytes()
            fo.seek(w_off + st * PACK)
            fo.write(pb)
            fo.seek(s_off + st * SCALE_COLS)
            fo.write(sb)
            done += c
            if (st // chunk_rows) % mon == 0 or done == rows:
                dt = time.time() - t0
                rps = done / dt if dt > 0 else 0
                eta = (rows - done) / rps if rps > 0 else 0
                log(
                    f"  {wname.split('.')[1]} rows {done}/{rows} ({100*done/rows:.1f}%) "
                    f"{rps/1e6:.2f} Mrows/s ETA {eta/60:.1f} min "
                    f"gpu {gpu_mem_gib(device):.1f} GiB rss {rss_gib():.1f} GiB "
                    f"free {free_gib('/SHARE'):.0f} GiB"
                )
    del w, s_u8, wf, scale, ref, packed, scale_u8
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def build_shard(src, out, sh, converted, device, chunk_rows, mon):
    """Write one output shard atomically. Returns (out_hash, out_bytes, entries)."""
    src_path = os.path.join(src, sh)
    out_path = os.path.join(out, sh)
    _, hdr_src, data_start_src = read_st_header(src_path)
    meta = hdr_src.get("__metadata__", {"format": "pt"})
    names = [k for k in hdr_src if k != "__metadata__"]
    if sh not in converted:
        # Pure-copy shard: duplicate the entire source file verbatim (header +
        # data) so it is byte-identical to source. Atomic .tmp + fsync + rename.
        tmp_path = f"{out_path}.tmp.{os.getpid()}"
        log(f"copying {sh} verbatim: {len(names)} tensors (byte-identical)")
        h = hashlib.sha256()
        with open(src_path, "rb") as fs, open(tmp_path, "wb") as fo:
            while True:
                b = fs.read(COPY_BLOCK)
                if not b:
                    break
                fo.write(b)
                h.update(b)
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(tmp_path, out_path)
        out_bytes = os.path.getsize(out_path)
        entries = {n: {"nbytes": hdr_src[n]["data_offsets"][1] - hdr_src[n]["data_offsets"][0],
                       "dtype": hdr_src[n]["dtype"], "shape": hdr_src[n]["shape"],
                       "out_off": hdr_src[n]["data_offsets"][0]} for n in names}
        log(f"  copied {sh}: {out_bytes} bytes sha256 {h.hexdigest()[:16]}…")
        return h.hexdigest(), out_bytes, entries
    out_entries = {}
    cur = 0
    header_json = {}
    for name in names:
        info = hdr_src[name]
        if name.endswith(CONVERT_SUFFIXES):
            od, osh, onb = converted_spec(name, info)
            out_entries[name] = dict(kind="convert", dtype=od, shape=osh, nbytes=onb)
        else:
            o0, o1 = info["data_offsets"]
            out_entries[name] = dict(
                kind="copy", dtype=info["dtype"], shape=info["shape"],
                nbytes=o1 - o0, src0=data_start_src + o0,
            )
        e = out_entries[name]
        e["out_off"] = cur
        header_json[name] = {"dtype": e["dtype"], "shape": e["shape"],
                             "data_offsets": [cur, cur + e["nbytes"]]}
        cur += e["nbytes"]
    header_json["__metadata__"] = meta
    hb = json.dumps(header_json, separators=(",", ":")).encode("utf-8")
    total_data = cur
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    log(f"building {sh}: {len(names)} tensors, {total_data} data bytes "
        f"({'convert' if sh in converted else 'copy'})")
    with open(tmp_path, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)))
        fo.write(hb)
        data0 = 8 + len(hb)
        fo.truncate(data0 + total_data)
        with open(src_path, "rb") as fs:
            for name in names:
                e = out_entries[name]
                if e["kind"] == "copy":
                    copy_range(fs, fo, e["src0"], e["nbytes"], data0 + e["out_off"])
        if sh in converted:
            convert_embed_pair(src_path, fo, data0, out_entries, device, chunk_rows, mon)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp_path, out_path)
    out_hash = sha256_file(out_path)
    out_bytes = os.path.getsize(out_path)
    log(f"  wrote {sh}: {out_bytes} bytes sha256 {out_hash[:16]}…")
    return out_hash, out_bytes, {n: out_entries[n] for n in names}


def shard_is_valid(out, src, sh):
    """Resume check: does an existing output shard match the expected contract?"""
    out_path = os.path.join(out, sh)
    src_path = os.path.join(src, sh)
    if not os.path.exists(out_path):
        return False
    try:
        _, hdr_out, _ = read_st_header(out_path)
        _, hdr_src, _ = read_st_header(src_path)
    except Exception:
        return False
    # Pure-copy shards must be byte-identical to source: enforce exact file size
    # (a verbatim copy has identical total size; the old header-rebuild variant
    # does not). Full byte-identity is confirmed later by --verify-output.
    if not any(n.endswith(CONVERT_SUFFIXES) for n in hdr_src if n != "__metadata__"):
        return os.path.getsize(out_path) == os.path.getsize(src_path)
    names_out = set(k for k in hdr_out if k != "__metadata__")
    names_src = set(k for k in hdr_src if k != "__metadata__")
    if names_out != names_src:
        return False
    for name in names_src:
        info_src = hdr_src[name]
        info_out = hdr_out[name]
        if name.endswith(CONVERT_SUFFIXES):
            od, osh, onb = converted_spec(name, info_src)
            if info_out["dtype"] != od or info_out["shape"] != osh:
                return False
        else:
            if info_out["dtype"] != info_src["dtype"] or info_out["shape"] != info_src["shape"]:
                return False
        o0, o1 = info_out["data_offsets"]
        exp = (onb if name.endswith(CONVERT_SUFFIXES) else dtype_nbytes_from_source(info_src))
        if o1 - o0 != exp:
            return False
    # size must cover header + data
    _, hb, data0 = read_st_header(out_path)
    total = max(hdr_out[n]["data_offsets"][1] for n in names_out)
    if os.path.getsize(out_path) != data0 + total:
        return False
    return True


def copy_ancillary(src, out):
    copied = []
    for item in ANCILLARY:
        sp = os.path.join(src, item)
        dp = os.path.join(out, item)
        if not os.path.exists(sp):
            continue
        if os.path.isdir(sp):
            if os.path.exists(dp):
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
        else:
            tmp = f"{dp}.tmp.{os.getpid()}"
            shutil.copy2(sp, tmp)
            os.replace(tmp, dp)
        copied.append(item)
    return copied


def write_config(src, out):
    cfg = json.load(open(os.path.join(src, CONFIG_NAME)))
    tc = cfg.setdefault("text_config", {})
    tc["engram_quant_scheme"] = SCHEME
    tc["engram_quant_group_size"] = INT2_GROUP
    tc["engram_quant_codebook_r"] = 0.3
    dp = os.path.join(out, CONFIG_NAME)
    tmp = f"{dp}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dp)


def write_index(out, scan, out_entries_all):
    """Rebuild model.safetensors.index.json from actual output tensor bytes."""
    wm = scan["weight_map"]
    total = 0
    for sh, entries in out_entries_all.items():
        for name, e in entries.items():
            total += e["nbytes"]
    idx = {"metadata": {"total_size": total}, "weight_map": dict(wm)}
    dp = os.path.join(out, INDEX_NAME)
    tmp = f"{dp}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dp)
    return total


def git_state(path):
    try:
        import subprocess
        h = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "-C", path, "diff", "--quiet"],
                                stderr=subprocess.DEVNULL) != 0
        return {"head": h, "dirty": dirty}
    except Exception:
        return {"head": None, "dirty": None}


def do_convert(args, scan):
    src, out = args.source, args.output
    os.makedirs(out, exist_ok=True)
    device = args.device
    convert_shards = set(scan["convert_shards"])
    manifest_path = os.path.join(out, MANIFEST_NAME)
    manifest = {
        "scheme": SCHEME, "version": "dsv41_engram_int2_v1",
        "packing": "4x2-bit LSB-first per byte", "group_size": INT2_GROUP,
        "codebook_r": 0.3, "codebook": [-1.0, -0.3, 0.3, 1.0], "dim": INT2_DIM,
        "source": os.path.abspath(src), "output": os.path.abspath(out),
        "source_config_sha256": sha256_file(os.path.join(src, CONFIG_NAME)),
        "source_index_sha256": sha256_file(os.path.join(src, INDEX_NAME)),
        "converted_tensors": sorted(scan["convert_tensors"].keys()),
        "preserved_tensor_count": len(scan["weight_map"]) - len(scan["convert_tensors"]),
        "converter_path": os.path.abspath(__file__),
        "sglang_module": sys.modules["sglang.kernels.ops.embeddings.engram_int2"].__file__,
        "sglang_git": git_state(os.environ.get("SGLANG_REPO", "/opt/sglang_dsv41_tp2_int2_20261007")),
        "device": device, "chunk_rows": args.chunk_rows,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shards": {}, "complete": False,
    }
    out_entries_all = {}
    for sh in scan["shards"]:
        if args.resume and shard_is_valid(out, src, sh):
            log(f"resume: {sh} already valid, skipping")
            _, hdr, data_start = read_st_header(os.path.join(out, sh))
            names = [k for k in hdr if k != "__metadata__"]
            out_entries_all[sh] = {
                n: {"nbytes": hdr[n]["data_offsets"][1] - hdr[n]["data_offsets"][0],
                    "dtype": hdr[n]["dtype"], "shape": hdr[n]["shape"],
                    "out_off": hdr[n]["data_offsets"][0]} for n in names}
            manifest["shards"][sh] = {
                "output_bytes": os.path.getsize(os.path.join(out, sh)),
                "output_sha256": sha256_file(os.path.join(out, sh)),
                "resumed": True, "converted": sh in convert_shards}
            continue
        if os.path.exists(os.path.join(out, sh)) and not args.resume:
            raise SystemExit(f"REFUSE to overwrite existing {sh} without --resume")
        if os.path.exists(os.path.join(out, sh)) and not shard_is_valid(out, src, sh):
            raise SystemExit(
                f"REFUSE to overwrite existing INVALID {sh}; inspect manually, do not auto-restart")
        oh, ob, entries = build_shard(src, out, sh, convert_shards, device, args.chunk_rows, args.monitor_every)
        out_entries_all[sh] = entries
        manifest["shards"][sh] = {
            "output_bytes": ob, "output_sha256": oh,
            "resumed": False, "converted": sh in convert_shards}
        # free-space guard
        fg = free_gib("/SHARE")
        if fg < 64:
            raise SystemExit(f"ABORT: /SHARE free {fg:.1f} GiB < 64 GiB floor")

    log("copying ancillary files…")
    manifest["ancillary_copied"] = copy_ancillary(src, out)
    log("writing config.json (INT2 scheme)…")
    write_config(src, out)
    log("writing rebuilt index…")
    total = write_index(out, scan, out_entries_all)
    manifest["output_total_size"] = total
    manifest["tensor_count"] = len(scan["weight_map"])
    manifest["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # manifest complete only set True by verify-output after all checks pass
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"conversion data written; manifest at {manifest_path} (complete=False until verify)")


def do_verify(args, scan):
    """Phase 4 artifact verification against the source checkpoint."""
    src, out = args.source, args.output
    problems = []
    ok = lambda c, m: (print(f"  [PASS] {m}") if c else (problems.append(m) or print(f"  [FAIL] {m}")))
    log("=== VERIFY OUTPUT ===")
    # 1. exactly N shards
    out_shards = sorted(f for f in os.listdir(out) if f.endswith(".safetensors"))
    ok(len(out_shards) == len(scan["shards"]), f"shard count {len(out_shards)} == {len(scan['shards'])}")
    # 2+3. index resolves + every shard opens
    idx = json.load(open(os.path.join(out, INDEX_NAME)))
    wm_out = idx["weight_map"]
    resolve_ok = all(os.path.exists(os.path.join(out, sh)) for sh in set(wm_out.values()))
    ok(resolve_ok, "every index entry resolves to an existing shard")
    opens_ok = True
    out_names = set()
    out_headers = {}
    for sh in out_shards:
        try:
            _, hdr, _ = read_st_header(os.path.join(out, sh))
            out_headers[sh] = hdr
            for n in hdr:
                if n != "__metadata__":
                    out_names.add(n)
        except Exception as e:
            opens_ok = False
            print(f"    open fail {sh}: {e}")
    ok(opens_ok, "every shard opens with a valid safetensors header")
    # 4+5. tensor name set + count
    src_names = set(scan["weight_map"].keys())
    ok(out_names == src_names, f"tensor-name set equals source ({len(out_names)} names)")
    ok(len(out_names) == len(src_names), f"tensor count {len(out_names)} == {len(src_names)}")
    # 6+7. only 4 converted contracts differ; shapes/dtypes exact
    conv = set(scan["convert_tensors"].keys())
    contract_ok = True
    for sh in scan["convert_shards"]:
        _, hdr_src, _ = read_st_header(os.path.join(src, sh))
        hdr_out = out_headers[sh]
        for name in [k for k in hdr_src if k != "__metadata__"]:
            si, oi = hdr_src[name], hdr_out[name]
            if name.endswith(CONVERT_SUFFIXES):
                od, osh, _ = converted_spec(name, si)
                if oi["dtype"] != od or oi["shape"] != osh:
                    contract_ok = False
                    print(f"    bad converted contract {name}: {oi['dtype']} {oi['shape']}")
            else:
                if oi["dtype"] != si["dtype"] or oi["shape"] != si["shape"]:
                    contract_ok = False
                    print(f"    converted-shard preserved tensor changed {name}")
    ok(contract_ok, "converted tensors match INT2 contract; preserved tensors in 47/48 unchanged shape/dtype")
    # 8. unchanged shards byte-identical (stream hash compare)
    unchanged = [s for s in scan["shards"] if s not in set(scan["convert_shards"])]
    bytewise_ok = True
    for sh in unchanged:
        hs = sha256_file(os.path.join(src, sh))
        ho = sha256_file(os.path.join(out, sh))
        if hs != ho:
            bytewise_ok = False
            print(f"    byte mismatch {sh}")
    ok(bytewise_ok, f"all {len(unchanged)} unchanged shards byte-identical to source")
    # 9. unchanged tensors in 47/48 byte-identical (range compare)
    pres_ok = True
    for sh in scan["convert_shards"]:
        _, hdr_src, ds_src = read_st_header(os.path.join(src, sh))
        _, hdr_out, ds_out = read_st_header(os.path.join(out, sh))
        with open(os.path.join(src, sh), "rb") as fs, open(os.path.join(out, sh), "rb") as fo:
            for name in [k for k in hdr_src if k != "__metadata__"]:
                if name.endswith(CONVERT_SUFFIXES):
                    continue
                so0, so1 = hdr_src[name]["data_offsets"]
                oo0, oo1 = hdr_out[name]["data_offsets"]
                hs = sha256_range(fs, ds_src + so0, so1 - so0)
                ho = sha256_range(fo, ds_out + oo0, oo1 - oo0)
                if hs != ho:
                    pres_ok = False
                    print(f"    preserved tensor mismatch {sh}:{name}")
    ok(pres_ok, "unchanged tensors in shards 47/48 byte-identical to source")
    # 10. index total == actual tensor bytes
    actual = 0
    for sh in out_shards:
        for n, info in out_headers[sh].items():
            if n == "__metadata__":
                continue
            o0, o1 = info["data_offsets"]
            actual += o1 - o0
    ok(idx["metadata"]["total_size"] == actual, f"index total_size {idx['metadata']['total_size']} == actual {actual}")
    # 11. no tmp/partial files
    tmps = [f for f in os.listdir(out) if ".tmp." in f]
    ok(len(tmps) == 0, "no .tmp/partial files remain")
    # 12. config declares scheme; source unchanged
    cfg = json.load(open(os.path.join(out, CONFIG_NAME)))
    tc = cfg.get("text_config", {})
    ok(tc.get("engram_quant_scheme") == SCHEME, "config declares INT2 scheme under text_config")
    ok(sha256_file(os.path.join(src, CONFIG_NAME)) != None, "source config readable")  # trivial
    # 13. sampled rows start/middle/end both layers: codes valid, kernel==oracle, NMSE/cosine
    numerics = {}
    sample_ok = True
    for sh in scan["convert_shards"]:
        _, hdr_out, _ = read_st_header(os.path.join(out, sh))
        wname = [k for k in hdr_out if k.endswith(".engram.embed.weight")][0]
        sname = wname[: -len(".weight")] + ".scale"
        layer = wname.split(".")[1]
        rows = hdr_out[wname]["shape"][0]
        starts = [0, rows // 2, rows - args.sample_rows]
        refs, deqs = [], []
        with safe_open(os.path.join(out, sh), framework="pt") as fo, \
             safe_open(os.path.join(src, sh), framework="pt") as fsrc:
            wsl = fo.get_slice(wname); ssl = fo.get_slice(sname)
            swsl = fsrc.get_slice(wname); sssl = fsrc.get_slice(sname)
            for st in starts:
                c = args.sample_rows
                packed = wsl[st:st+c].to(args.device)
                scl = ssl[st:st+c].to(args.device)
                kout = torch.empty(c, INT2_DIM, dtype=torch.bfloat16, device=args.device)
                ids = torch.arange(st, st+c, device=args.device, dtype=torch.int64)
                engram_int2_gather(packed.data_ptr(), scl.data_ptr(), ids,
                                   kout.view(-1, INT2_DIM), INT2_DIM, INT2_GROUP,
                                   DEFAULT_R, row_lo=st, row_hi=st+c)
                oracle = dequant_int2_torch(packed, scl, DEFAULT_R, INT2_DIM, INT2_GROUP)
                maxdiff = (kout.float() - oracle.float()).abs().max().item()
                if maxdiff != 0.0:
                    sample_ok = False
                    print(f"    kernel!=oracle layer{layer} st{st} maxdiff {maxdiff}")
                if not torch.isfinite(kout.float()).all():
                    sample_ok = False
                # source ref for NMSE
                sw = swsl[st:st+c].to(args.device).float()
                ss = sssl[st:st+c].view(torch.uint8).to(args.device)
                scale = torch.exp2(ss.to(torch.int32).float() - 127.0)
                ref = (sw.view(c, INT2_DIM//INT2_GROUP, INT2_GROUP) * scale.unsqueeze(-1)).reshape(c, INT2_DIM)
                refs.append(ref); deqs.append(oracle.float())
        ref = torch.cat(refs); deq = torch.cat(deqs)
        nmse = ((deq-ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-30)).item()
        cos = torch.nn.functional.cosine_similarity(deq, ref, dim=-1).mean().item()
        numerics[f"layer{layer}"] = {"nmse": nmse, "cosine": cos}
        print(f"    layer{layer} sampled NMSE {nmse*100:.3f}% cosine {cos:.4f}")
    ok(sample_ok, "sampled rows: valid codes, no NaN/Inf, kernel==oracle bitwise")
    # 14. free space floor
    fg = free_gib("/SHARE")
    ok(fg >= 64, f"/SHARE free {fg:.1f} GiB >= 64 GiB")
    # finalize manifest
    mpath = os.path.join(out, MANIFEST_NAME)
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    else:
        manifest = {}
    manifest["verification"] = {
        "passed": len(problems) == 0, "problems": problems,
        "numerics": numerics, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_free_gib": fg}
    manifest["complete"] = len(problems) == 0
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    if problems:
        print(f"\nVERIFY FAILED: {len(problems)} problem(s): {problems}")
        return 1
    print("\nVERIFY PASSED: all artifact checks green; manifest complete=True")
    return 0


def do_smoke(args, scan):
    """Phase 2 dual-layer sample smoke: GPU quantize/pack, save/reopen, kernel vs
    oracle, NMSE/cosine, no NaN/Inf. Writes a tiny artifact outside the destination."""
    src = args.source
    device = args.device
    blocks, block_rows = 5, 4096
    smoke_dir = args.smoke_dir
    os.makedirs(smoke_dir, exist_ok=True)
    report = {}
    all_ref, all_deq = [], []
    for sh in scan["convert_shards"]:
        _, hdr, _ = read_st_header(os.path.join(src, sh))
        wname = [k for k in hdr if k.endswith(".engram.embed.weight")][0]
        sname = wname[: -len(".weight")] + ".scale"
        layer = wname.split(".")[1]
        rows = hdr[wname]["shape"][0]
        starts = [int(i * (rows - block_rows) / (blocks - 1)) for i in range(blocks)]
        packs, scales, refs = [], [], []
        with safe_open(os.path.join(src, sh), framework="pt") as f:
            wsl = f.get_slice(wname); ssl = f.get_slice(sname)
            for st in starts:
                w = wsl[st:st+block_rows].to(device)
                s = ssl[st:st+block_rows].view(torch.uint8).to(device)
                wf = w.float()
                scale = torch.exp2(s.to(torch.int32).float() - 127.0)
                ref = (wf.view(block_rows, INT2_DIM//INT2_GROUP, INT2_GROUP) * scale.unsqueeze(-1)).reshape(block_rows, INT2_DIM)
                packed, scale_u8 = quantize_int2(ref)
                packs.append(packed); scales.append(scale_u8); refs.append(ref)
        packed = torch.cat(packs); scale_u8 = torch.cat(scales); ref = torch.cat(refs)
        # save/reopen safetensors round-trip (byte-exact)
        from safetensors.torch import save_file, load_file
        sp = os.path.join(smoke_dir, f"smoke_layer{layer}.safetensors")
        save_file({"weight": packed.cpu(), "scale": scale_u8.cpu()}, sp)
        rl = load_file(sp)
        reload_ok = torch.equal(rl["weight"], packed.cpu()) and torch.equal(rl["scale"], scale_u8.cpu())
        # kernel vs oracle on reloaded bytes
        pk = rl["weight"].to(device); sk = rl["scale"].to(device)
        n = pk.shape[0]
        out = torch.empty(n, INT2_DIM, dtype=torch.bfloat16, device=device)
        ids = torch.arange(n, device=device, dtype=torch.int64)
        engram_int2_gather(pk.data_ptr(), sk.data_ptr(), ids, out.view(-1, INT2_DIM),
                           INT2_DIM, INT2_GROUP, DEFAULT_R, row_lo=0, row_hi=n)
        oracle = dequant_int2_torch(pk, sk, DEFAULT_R, INT2_DIM, INT2_GROUP)
        maxdiff = (out.float() - oracle.float()).abs().max().item()
        finite = bool(torch.isfinite(out.float()).all())
        nmse = ((oracle.float()-ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-30)).item()
        cos = torch.nn.functional.cosine_similarity(oracle.float(), ref, dim=-1).mean().item()
        report[f"layer{layer}"] = {
            "sampled_rows": n, "reload_byte_exact": reload_ok,
            "kernel_oracle_maxdiff": maxdiff, "finite": finite,
            "nmse": nmse, "cosine": cos}
        all_ref.append(ref); all_deq.append(oracle.float())
        print(f"  layer{layer}: reload_exact={reload_ok} kernel==oracle(maxdiff={maxdiff}) "
              f"finite={finite} NMSE={nmse*100:.3f}% cosine={cos:.4f}")
        os.remove(sp)
    ref = torch.cat(all_ref); deq = torch.cat(all_deq)
    comb_nmse = ((deq-ref).pow(2).sum()/ref.pow(2).sum().clamp_min(1e-30)).item()
    comb_cos = torch.nn.functional.cosine_similarity(deq, ref, dim=-1).mean().item()
    report["combined"] = {"nmse": comb_nmse, "cosine": comb_cos,
                          "calib_nmse": 0.1265, "calib_cosine": 0.936}
    passed = (all(r.get("reload_byte_exact", False) for k, r in report.items() if k.startswith("layer"))
              and all(r.get("kernel_oracle_maxdiff", 1) == 0.0 for k, r in report.items() if k.startswith("layer"))
              and all(r.get("finite", False) for k, r in report.items() if k.startswith("layer"))
              and abs(comb_nmse - 0.1265) < 0.02 and comb_cos > 0.90)
    report["passed"] = passed
    json.dump(report, open(os.path.join(smoke_dir, "smoke_report.json"), "w"), indent=2)
    print(f"\ncombined NMSE {comb_nmse*100:.3f}% (calib 12.65%) cosine {comb_cos:.4f} (calib 0.936)")
    print("SMOKE", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description="DSV4.1-Flash FP4PTPC -> Engram INT2 converter")
    ap.add_argument("--source", required=True, help="source checkpoint directory (read-only)")
    ap.add_argument("--output", help="destination checkpoint directory")
    ap.add_argument("--resume", action="store_true", help="validate+skip completed shards")
    ap.add_argument("--dry-run", action="store_true", help="report layout, create nothing")
    ap.add_argument("--smoke", action="store_true", help="dual-layer sample smoke conversion")
    ap.add_argument("--smoke-dir", default="/tmp/engram_int2_smoke", help="smoke artifact dir")
    ap.add_argument("--verify-output", action="store_true", help="verify an existing output")
    ap.add_argument("--device", default="cuda:0", help="conversion device")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000, help="GPU rows per chunk")
    ap.add_argument("--torch-threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--sample-rows", type=int, default=4096, help="verify sample rows per point")
    ap.add_argument("--monitor-every", type=int, default=8, help="log every N chunks")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    if args.dry_run or args.smoke or (not args.verify_output):
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            if not args.dry_run:
                raise SystemExit("CUDA/ROCm device requested but not available")
    scan = scan_source(args.source)

    if args.dry_run:
        do_dry_run(args.source, args.output, scan)
        return 0
    if args.smoke:
        return do_smoke(args, scan)
    if args.verify_output:
        if not args.output:
            raise SystemExit("--verify-output requires --output")
        return do_verify(args, scan)
    # full conversion
    if not args.output:
        raise SystemExit("conversion requires --output")
    do_convert(args, scan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
