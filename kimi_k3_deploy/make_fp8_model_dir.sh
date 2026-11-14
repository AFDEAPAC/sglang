#!/bin/bash
# Model dir for online fp8 dense layers: the original Kimi-K3 checkpoint (symlinked) plus a
# config.json whose quantization_config gains linear_fp8_config, so SGLang converts the dense
# layers not in ignored_layers to fp8 at load time.
#   make_fp8_model_dir.sh <original checkpoint dir> <new dir>
set -euo pipefail
SRC="${1:?original checkpoint dir}"
DST="${2:?new dir}"
mkdir -p "$DST"
for f in "$SRC"/* "$SRC"/.[!.]*; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ "$b" = config.json ] || ln -sfn "$f" "$DST/$b"
done
python3 - "$SRC/config.json" "$DST/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["text_config"]["quantization_config"]["linear_fp8_config"] = {
    "quant_method": "compressed-tensors-linear-fp8-online", "activation_scheme": "dynamic", "fmt": "e4m3",
    "ignored_layers": [
        "self_attn.kv_b_proj", "self_attn.fused_qkvg_proj", "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.g_proj", "self_attn.g_a_proj", "self_attn.g_b_proj",
        "self_attn.f_a_proj", "self_attn.f_b_proj", "self_attn.b_proj", "self_attn.qkv_conv1d",
        "self_attn.fused_qkvbfg_a_proj", "shared_experts.down_proj", "vision_tower", "mm_projector",
        "routed_expert_up_proj", "routed_expert_down_proj", "mlp.gate",
        "layers.90.mlp.shared_experts.gate_proj", "layers.90.mlp.shared_experts.up_proj",
        "layers.90.mlp.shared_experts.down_proj", "layers.91.mlp.shared_experts.gate_proj",
        "layers.91.mlp.shared_experts.up_proj", "layers.91.mlp.shared_experts.down_proj",
        "layers.92.mlp.shared_experts.gate_proj", "layers.92.mlp.shared_experts.up_proj",
        "layers.92.mlp.shared_experts.down_proj", "layers.12.self_attn.o_proj", "layers.92.self_attn.o_proj"]}
json.dump(c, open(sys.argv[2], "w"), indent=4)
PY
echo "wrote $DST"
