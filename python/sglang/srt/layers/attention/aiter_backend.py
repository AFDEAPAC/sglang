from __future__ import annotations

"""
end to end attention solution with aiter kernels
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_func,
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
        get_ps_metadata_info_v1,
        get_ps_metadata_v1,
        mha_batch_prefill_func,
        mla_prefill_ps_asm_fwd,
        mla_reduce_v1,
        paged_attention_ragged,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    from aiter.jit.utils.chip_info import get_gfx as _aiter_get_gfx
    from aiter.ops.triton.attention.unified_attention import unified_attention
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.utils import (
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

# Use aiter mla persist design for fp8-kv cache
_use_mla_ps_kernel = get_bool_env_var("SGLANG_AITER_MLA_PERSIST", "True")

# Fold qlen x nhead for MLA decode to skip aiter's nhead=8 -> 16 padding.
# MLA attention is per-head independent against a shared KV latent, so
# [bs*qlen, nhead, D] can be reshaped to [bs, qlen*nhead, D] with
# max_seqlen_q=1 when every request has a uniform qlen. This trades the
# implicit causal mask across qlen positions for extra bookkeeping, so it
# is only safe when the caller is OK with non-causal semantics across the
# folded qlen dimension (EAGLE3 target-verify / draft-extend uses full
# prefix context across the small qlen).
_mla_fold_enabled = get_bool_env_var("SGLANG_MLA_FOLD", "False")


def _try_fold_mla_q(q, tp_q_head_num, qk_head_dim, max_q_len, qo_indptr, kv_dtype=None):
    """If eligible, return a dict describing the folded Q/indptr, else None.

    Eligibility:
      - SGLANG_MLA_FOLD=1
      - tp_q_head_num == 8 (Kimi-K2.5 TP=8 setting; fold target is 16 or 32)
      - max_q_len in (2, 3, 4)
      - q.shape[0] evenly divisible by max_q_len (uniform qlen per batch)
      - not currently capturing a CUDA graph (ctypes-based persistent path
        may interact badly; be conservative)
    """
    if not _mla_fold_enabled:
        return None
    # aiter only has optimized persistent/ASM kernels for bf16/fp16 KV on folded
    # shapes (nhead=16/32, qseqlen=1). fp8 KV falls through to a heuristic miss
    # and aborts. Gate to avoid that crash.
    if kv_dtype is not None and kv_dtype not in (torch.bfloat16, torch.float16):
        return None
    if tp_q_head_num != 8:
        return None
    if max_q_len not in (2, 3, 4):
        return None
    total_s = q.shape[0]
    if total_s % max_q_len != 0:
        return None
    try:
        if torch.cuda.is_current_stream_capturing():
            return None
    except AttributeError:
        pass

    bs = total_s // max_q_len
    pad_qlen = max_q_len
    q_in = q
    if max_q_len == 3:
        # nhead=24 is not supported by aiter; pad qlen 3 -> 4, nhead -> 32.
        pad_qlen = 4
        q_padded = torch.zeros(
            bs * 4, tp_q_head_num, qk_head_dim, dtype=q.dtype, device=q.device
        )
        q_padded.view(bs, 4, tp_q_head_num, qk_head_dim)[:, :3].copy_(
            q.view(bs, 3, tp_q_head_num, qk_head_dim)
        )
        q_in = q_padded

    fold_nhead = pad_qlen * tp_q_head_num  # 16 or 32
    if fold_nhead not in (16, 32):
        return None

    q_folded = q_in.reshape(bs, fold_nhead, qk_head_dim).contiguous()
    qo_indptr_folded = torch.arange(
        bs + 1, dtype=qo_indptr.dtype, device=qo_indptr.device
    )
    return {
        "q": q_folded,
        "bs": bs,
        "fold_nhead": fold_nhead,
        "qo_indptr": qo_indptr_folded,
        "max_q_len": 1,
        "pad_qlen": pad_qlen,
        "orig_max_q_len": max_q_len,
    }


def _unfold_mla_o(o_fold, fold_info, tp_q_head_num, v_head_dim):
    """Reshape folded output [bs, pad_qlen*nhead, VD] back to
    [bs*orig_max_q_len, nhead, VD], slicing off the qlen padding if any."""
    bs = fold_info["bs"]
    pad_qlen = fold_info["pad_qlen"]
    orig = fold_info["orig_max_q_len"]
    o = o_fold.view(bs, pad_qlen, tp_q_head_num, v_head_dim)
    if pad_qlen != orig:
        o = o[:, :orig]
    return o.reshape(bs * orig, tp_q_head_num, v_head_dim).contiguous()

# Use fp8 prefill only on gfx95
_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and is_gfx95_supported()
)

# Persist
# fast_mode=True if _use_mla_ps_kernel else False
# intra_batch_mode=False if _use_mla_ps_kernel else True

# fake non-ps, intra_batch_mode needs to be True for non-ps-mode
fast_mode = False
intra_batch_mode = True if _use_mla_ps_kernel else False


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class ForwardMetadata:
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    max_q_len: int
    max_kv_len: Optional[int]
    work_metadata: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None
    num_kv_splits: Optional[int] = None
    run_graph: Optional[bool] = True
    custom_mask: Optional[torch.Tensor] = None
    mask_indptr: Optional[torch.Tensor] = None
    max_extend_len: Optional[int] = None
    fp8_prefill_kv_indices: Optional[torch.Tensor] = None
    swa_page_table: Optional[torch.Tensor] = None
    splitter_active: bool = False
    splitter_qlen: int = 0
    split_qo_indptr: Optional[torch.Tensor] = None
    split_kv_indptr_a: Optional[torch.Tensor] = None
    split_kv_indices_a: Optional[torch.Tensor] = None


global_workspace_buffer = None

_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        topk: int = 1,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        self.input_dtype = model_runner.model_config.dtype

        self.page_size = model_runner.server_args.page_size

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = topk
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA

        # Get v_head_dim based on model type
        if self.use_mla:
            # For MLA models, get v_head_dim from model config
            self.v_head_dim = model_runner.model_config.v_head_dim
            # PR 2852 splitter: record aiter-MLA effective dims.
            # aiter.mla_decode_fwd operates in kv_lora_rank output space.
            _mc = model_runner.model_config
            self._mla_qk_head_dim = _mc.kv_lora_rank + _mc.qk_rope_head_dim
            self._mla_v_head_dim = _mc.kv_lora_rank
        elif hasattr(model_runner.token_to_kv_pool, "get_v_head_dim"):
            # For hybrid models (Mamba+attention, GDN, Kimi linear),
            # layer_id=0 may not be a full attention layer
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int64, device=model_runner.device
        )
        self._kv_indices_scratch: Optional[torch.Tensor] = None

        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # sliding window attention
        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )

        if self.use_sliding_window_kv_pool:
            self.token_to_kv_pool = model_runner.token_to_kv_pool
            self.use_triton_unified_attention = True
        else:
            self.use_triton_unified_attention = get_bool_env_var(
                "SGLANG_USE_AITER_UNIFIED_ATTN"
            )

        # aiter kernel related initialization
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not (self.use_mla or self.use_triton_unified_attention):
            self.workspace_buffer = torch.empty(
                (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * nbyes_per_qo_elem
                + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
                device=self.device,
            )

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            self.enable_dp_attention = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            # current mla_decode_fwd onln support fake-nps in self.num_head == 16
            # so all num_head size does not use qh16 kernel to simulate
            # it should not use fake-nps (fast_mode = False, intra_batch_mode = True)
            # it will cause gpu-fault or accuracy issue
            if self.num_head == 32 or self.num_head == 128:
                fast_mode = True
                intra_batch_mode = False

            # nhead=8 (TP8 with 64 attention heads) needs padding to 16
            # for metadata and kernel calls
            self._mla_padded_nhead = self.num_head
            if self.num_head == 8:
                self._mla_padded_nhead = 16
            # Persistent padding buffers so aiter's forced_head_pad (zeros+alloc+slice)
            # becomes a no-op. Lazy-allocated on first use; reused across layers/steps.
            self._mla_pad_q_buf = None
            self._mla_pad_o_buf = None

            # PR 2852: nh=8 bf16 on gfx942 has a native PS kernel
            # (mla_dec_stage1_bf16_a16w16_subQ16_mqa16_ps).
            # When active, we keep PS mode ON for nh=8 bf16 and dynamically
            # build nh=8 metadata for qlen=2 via _resolve_mla_nhead().
            self._nh8_native_ps = (
                self.num_head == 8
                and self.kv_cache_dtype is not fp8_dtype
                and _aiter_get_gfx() == "gfx942"
            )
            if self._nh8_native_ps:
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
            # Exclude nh=8 bf16 gfx942 from the fallback -- stays on PS for PR 2852 path.
            if (
                self.num_head in (16, 128)
                or (self.num_head == 8 and not self._nh8_native_ps)
            ) and self.kv_cache_dtype is not fp8_dtype:
                _use_mla_ps_kernel = False
                fast_mode = False
                intra_batch_mode = False

            self.max_split_per_batch = 32 if _use_mla_ps_kernel else None

            if self.num_draft_tokens is None and _use_mla_ps_kernel:
                self.max_split_per_batch = 64

            self.fix_max_split_per_batch = self.max_split_per_batch

    def _pad_mla_q_8_to_16(self, q, total_s, qk_head_dim, v_head_dim):
        """Pre-pad q from [total_s, 8, qk_head_dim] to [total_s, 16, qk_head_dim] via a
        persistent buffer. Returns (q_padded, o_padded) where o_padded is an uninitialized
        [total_s, 16, v_head_dim] buffer. The last 8 query heads are zero-initialized once
        (buffer init), so subsequent calls only copy the 8 real heads.

        Caller is responsible for slicing the output back:
            real_o.copy_(o_padded[:, :8, :])
        """
        buf_q = self._mla_pad_q_buf
        if (
            buf_q is None
            or buf_q.dtype != q.dtype
            or buf_q.shape[0] < total_s
            or buf_q.shape[2] != qk_head_dim
        ):
            capacity = max(total_s, 2048)
            self._mla_pad_q_buf = torch.zeros(
                (capacity, 16, qk_head_dim), dtype=q.dtype, device=q.device
            )
        buf_o = self._mla_pad_o_buf
        if (
            buf_o is None
            or buf_o.dtype != self.input_dtype
            or buf_o.shape[0] < total_s
            or buf_o.shape[2] != v_head_dim
        ):
            capacity = max(total_s, 2048)
            self._mla_pad_o_buf = torch.empty(
                (capacity, 16, v_head_dim), dtype=self.input_dtype, device=self.device
            )
        q_padded = self._mla_pad_q_buf[:total_s]
        o_padded = self._mla_pad_o_buf[:total_s]
        q_padded[:, :8, :] = q
        return q_padded, o_padded

    def _resolve_mla_nhead(self, max_seqlen_qo):
        """Effective nhead for MLA metadata.

        PR 2852 native kernel supports (gfx942, nh=8, bf16, qlen=2) without
        padding. For any other max_seqlen_qo (including qlen=1 decode, qlen=4
        verify with draft=4), we fall back to the nh-padded=16 path which
        hands off to AITER's forced_head_pad internally.
        """
        if getattr(self, "_nh8_native_ps", False) and max_seqlen_qo == 2:
            return 8
        return self._mla_padded_nhead

    def _populate_splitter_metadata(self, bs, seq_lens, req_pool_indices):
        """Build qlen=2 sub-call A metadata (kv ends at seq+2).
        Call B reuses main kv_indptr/kv_indices."""
        split = 2
        split_qo_indptr = self.cg_split_qo_indptr[: bs + 1]
        split_qo_indptr[: bs + 1] = torch.arange(
            0, (1 + bs) * split, step=split,
            dtype=torch.int32, device=self.device,
        )
        kv_lens_a = seq_lens + split
        split_kv_indptr_a = self.cg_split_kv_indptr_a[: bs + 1]
        split_kv_indptr_a[0] = 0
        split_kv_indptr_a[1 : bs + 1] = torch.cumsum(kv_lens_a, dim=0)
        split_kv_indices_a = self.cg_split_kv_indices_a
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            kv_lens_a,
            split_kv_indptr_a,
            None,
            split_kv_indices_a,
            self.req_to_token.stride(0),
        )
        return split_qo_indptr, split_kv_indptr_a, split_kv_indices_a

    def _run_mla_splitter(self, q, K_Buffer, o, layer, k_descale, intra_batch_mode):
        """PR 2852 splitter dispatch.

        qlen_total:
          1 -> pad q[bs,1,nh,qhd] to [bs,2,nh,qhd] (dummy at pos 1), single
               kernel call reusing main kv_indptr/kv_indices, return out[:,0].
          3 -> pad qlen 3->4 (dummy at pos 3), split to 2 x qlen=2 (calls A,B).
          4 -> 2 x qlen=2 split (calls A,B).
        """
        from aiter.mla import mla_decode_fwd
        fm = self.forward_metadata
        qlen_total = fm.splitter_qlen
        assert qlen_total in (1, 3, 4), f"splitter expects qlen in {{1,3,4}}, got {qlen_total}"
        bs = fm.split_qo_indptr.numel() - 1
        nh = q.shape[1]
        qhd = q.shape[2]
        vhd = o.shape[2]

        out_a = self.cg_split_out_a[: bs * 2]
        num_kv_splits = fm.num_kv_splits

        if qlen_total == 1:
            # Pad q[bs,1,nh,qhd] -> qa[bs,2,nh,qhd] with [dummy, real].
            # qlen=2 kernel causal aligns position 1 with last kv slot (= seq_len-1,
            # i.e. own-token K/V); position 0 aligns with kv[seq_len-2] (misses
            # last slot). Real q must go at position 1 for its window to cover
            # kv[0..seq_len-1] including its own K/V.
            qa = self.cg_split_qa[: bs * 2].view(bs, 2, nh, qhd)
            qa[:, 0].zero_()
            qa[:, 1].copy_(q.view(bs, nh, qhd))

            # Rebuild metadata for qlen=2, reusing main kv_indptr/kv_indices
            # (no split — real token at pos 0, dummy at pos 1 sees the same KV).
            self.make_mla_meta_data(
                fm.split_qo_indptr,
                fm.kv_indptr,
                fm.kv_last_page_len,
                self.work_metadata, self.work_info_set, self.work_indptr,
                self.reduce_indptr, self.reduce_final_map, self.reduce_partial_map,
                2,
                fast_mode=True,
                max_split_per_batch=num_kv_splits,
                intra_batch_mode=intra_batch_mode,
            )
            mla_decode_fwd(
                self.cg_split_qa[: bs * 2],
                K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                out_a,
                fm.split_qo_indptr,
                fm.kv_indptr,
                fm.kv_indices,
                fm.kv_last_page_len,
                2,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=self.work_metadata,
                work_indptr=self.work_indptr,
                work_info_set=self.work_info_set,
                reduce_indptr=self.reduce_indptr,
                reduce_final_map=self.reduce_final_map,
                reduce_partial_map=self.reduce_partial_map,
                q_scale=k_descale, kv_scale=k_descale,
                intra_batch_mode=intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
            o.view(bs, nh, vhd).copy_(out_a.view(bs, 2, nh, vhd)[:, 1])
            return o

        qa = self.cg_split_qa[: bs * 2].view(bs, 2, nh, qhd)
        qb = self.cg_split_qb[: bs * 2].view(bs, 2, nh, qhd)
        out_b = self.cg_split_out_b[: bs * 2]

        q4 = q.view(bs, qlen_total, nh, qhd)
        qa.copy_(q4[:, 0:2])
        if qlen_total == 4:
            qb.copy_(q4[:, 2:4])
        else:  # qlen=3: [dummy=0, q[2]]
            # cg_split_qb was initialized to 0; overwrite index 1 with real q[2].
            qb[:, 0].zero_()
            qb[:, 1].copy_(q4[:, 2])
        # Populate + launch call A
        self.make_mla_meta_data(
            fm.split_qo_indptr,
            fm.split_kv_indptr_a,
            fm.kv_last_page_len,
            self.work_metadata, self.work_info_set, self.work_indptr,
            self.reduce_indptr, self.reduce_final_map, self.reduce_partial_map,
            2,
            fast_mode=True,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=intra_batch_mode,
        )
        mla_decode_fwd(
            self.cg_split_qa[: bs * 2],
            K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
            out_a,
            fm.split_qo_indptr,
            fm.split_kv_indptr_a,
            fm.split_kv_indices_a,
            fm.kv_last_page_len,
            2,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            work_meta_data=self.work_metadata,
            work_indptr=self.work_indptr,
            work_info_set=self.work_info_set,
            reduce_indptr=self.reduce_indptr,
            reduce_final_map=self.reduce_final_map,
            reduce_partial_map=self.reduce_partial_map,
            q_scale=k_descale, kv_scale=k_descale,
            intra_batch_mode=intra_batch_mode,
            num_kv_splits=num_kv_splits,
        )

        # Populate + launch call B (reuses main kv_indptr/kv_indices for full kv range).
        self.make_mla_meta_data(
            fm.split_qo_indptr,
            fm.kv_indptr,
            fm.kv_last_page_len,
            self.work_metadata, self.work_info_set, self.work_indptr,
            self.reduce_indptr, self.reduce_final_map, self.reduce_partial_map,
            2,
            fast_mode=True,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=intra_batch_mode,
        )
        mla_decode_fwd(
            self.cg_split_qb[: bs * 2],
            K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
            out_b,
            fm.split_qo_indptr,
            fm.kv_indptr,
            fm.kv_indices,
            fm.kv_last_page_len,
            2,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
            work_meta_data=self.work_metadata,
            work_indptr=self.work_indptr,
            work_info_set=self.work_info_set,
            reduce_indptr=self.reduce_indptr,
            reduce_final_map=self.reduce_final_map,
            reduce_partial_map=self.reduce_partial_map,
            q_scale=k_descale, kv_scale=k_descale,
            intra_batch_mode=intra_batch_mode,
            num_kv_splits=num_kv_splits,
        )

        # Stitch outputs back into o.
        o4 = o.view(bs, qlen_total, nh, vhd)
        o4[:, 0:2].copy_(out_a.view(bs, 2, nh, vhd))
        if qlen_total == 4:
            o4[:, 2:4].copy_(out_b.view(bs, 2, nh, vhd))
        else:  # qlen=3: discard dummy at position 0, keep real at position 1
            o4[:, 2:3].copy_(out_b.view(bs, 2, nh, vhd)[:, 1:2])
        return o

    def make_mla_decode_meta_data_buffer(self, max_seqlen_qo, batch_size):
        # PR 2852 splitter: when _nh8_native_ps is on, ALL qlen traffic
        # (1, 3, 4) is routed through the qlen=2 kernel via the splitter.
        # Buffer must be sized for qlen=2 metadata.  qlen=2 itself (d=2)
        # stays at max_seqlen_qo=2 naturally.
        if getattr(self, "_nh8_native_ps", False):
            max_seqlen_qo = 2
        nhead = self._resolve_mla_nhead(max_seqlen_qo)
        dtype = self.kv_cache_dtype

        if self.enable_dp_attention:
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            self.max_split_per_batch = min(
                (cu_num + batch_size - 1) // batch_size, self.fix_max_split_per_batch
            )

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            batch_size,
            max_seqlen_qo,
            nhead,
            dtype,
            dtype,
            is_sparse=False,
            fast_mode=fast_mode,
            num_kv_splits=self.max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
        )

        # aiter implementation
        # the tensor's meaning please refer aiter/ops/attention.py
        work_metadata = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device="cuda"
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device="cuda"
        )
        work_info_set = torch.empty(
            work_info_set_size,
            dtype=work_info_set_type,
            device="cuda",
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
        )

        return (
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_meta_data(
        self,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_q_len,
        fast_mode,
        max_split_per_batch,
        intra_batch_mode,
    ):

        nhead_kv = 1
        page_size = self.page_size
        dtype = self.kv_cache_dtype

        effective_nhead = self._resolve_mla_nhead(max_q_len)
        meta = get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            effective_nhead // nhead_kv,
            nhead_kv,
            False,
            work_metadata,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(page_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=fast_mode,
            max_split_per_batch=max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
            dtype_q=dtype,
            dtype_kv=dtype,
        )

    def make_mla_prefill_ps_meta_data_buffer(
        self, batch_size: int, max_qlen: int, qlen_granularity: int
    ):
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_size, work_info_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_ps_metadata_info_v1(
            batch_size=batch_size,
            num_head_k=self.num_kv_head,
            max_qlen=max_qlen,
            qlen_granularity=qlen_granularity,
        )

        device = self.device
        work_metadata_ptrs = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        )

        return (
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_prefill_ps_meta_data(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
        work_metadata: torch.Tensor,
        work_indptr: torch.Tensor,
        work_info: torch.Tensor,
        reduce_indptr: torch.Tensor,
        reduce_final_map: torch.Tensor,
        reduce_partial_map: torch.Tensor,
        is_causal: bool = True,
    ):
        gqa_ratio = self.num_head // self.num_kv_head
        num_heads_k = self.num_kv_head
        tile_q = 256
        qhead_granularity = gqa_ratio
        qlen_granularity = tile_q // qhead_granularity
        kvlen_granularity = max(128, self.page_size)
        block_size = self.page_size

        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        kv_indptr_cpu = kv_indptr.to("cpu", dtype=torch.int32)
        seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_heads_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=is_causal,
        )

    # for page size > 1 useful conversion function
    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def _resolve_v2_num_draft_tokens(
        self,
        extend_seq_lens: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[list[int]] = None,
    ) -> int:
        """Resolve fixed per-request extend length for DRAFT_EXTEND_V2."""
        num_draft_tokens = self.num_draft_tokens
        if num_draft_tokens is None:
            # Prefer CPU-side metadata to avoid a GPU->CPU sync stall.
            if extend_seq_lens_cpu:
                num_draft_tokens = max(extend_seq_lens_cpu)
            elif extend_seq_lens is not None and extend_seq_lens.numel() > 0:
                num_draft_tokens = int(extend_seq_lens[0].item())
            else:
                raise ValueError(
                    "DRAFT_EXTEND_V2 requires speculative_num_draft_tokens or "
                    "non-empty extend_seq_lens/extend_seq_lens_cpu."
                )

        num_draft_tokens = int(num_draft_tokens)
        if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
            if not torch.all(extend_seq_lens == num_draft_tokens):
                raise ValueError(
                    "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                    f"extend_seq_lens={extend_seq_lens}, expected all == {num_draft_tokens}."
                )
        if extend_seq_lens_cpu and any(
            x != num_draft_tokens for x in extend_seq_lens_cpu
        ):
            raise ValueError(
                "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                f"{extend_seq_lens_cpu}, expected all == {num_draft_tokens}."
            )
        return num_draft_tokens

    def _get_kv_indices_scratch(
        self, required_tokens: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._kv_indices_scratch is None
            or self._kv_indices_scratch.device != device
            or self._kv_indices_scratch.numel() < required_tokens
        ):
            self._kv_indices_scratch = torch.empty(
                required_tokens, dtype=torch.int32, device=device
            )
        return self._kv_indices_scratch[:required_tokens]

    def _set_uniform_qo_indptr(
        self, bs: int, tokens_per_req: int, device: torch.device
    ) -> torch.Tensor:
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * tokens_per_req + 1,
            step=tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        return qo_indptr

    def _ensure_spec_v2_topk_supported(self):
        if self.topk > 1:
            raise NotImplementedError(
                "AiterAttnBackend SPEC_V2 path currently supports topk <= 1 only. "
                f"Got topk={self.topk}."
            )

    def mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
    ):
        total_q = q.shape[0]
        nhead = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim

        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)
        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        tile_q = 256
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map

        logits = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty(
            (total_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        output = q.new_empty(
            (total_q, nhead, v_head_dim),
            dtype=self.input_dtype,
        )

        mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.fp8_prefill_kv_indices,
            self.forward_metadata.work_indptr,
            self.forward_metadata.work_info_set,
            self.forward_metadata.max_q_len,
            layer.scaling,
            True,
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )
        mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )
        return output

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for aiter attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None
        max_kv_len = None

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        num_kv_splits = None
        swa_page_table = None
        max_kv_len = forward_batch.seq_lens_cpu.max().item()

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or forward_batch.forward_mode.is_idle():
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                if not self.use_triton_unified_attention:
                    kv_indices = self._get_kv_indices_scratch(
                        forward_batch.seq_lens_sum, forward_batch.seq_lens.device
                    )
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    max_q_len = 1
                    page_size = self.page_size
                    max_num_blocks_per_seq = (max_kv_len + page_size - 1) // page_size
                    kv_indices = torch.zeros(
                        bs, max_kv_len, dtype=torch.int32, device=self.device
                    )

                    create_flashmla_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                        max_kv_len,
                        1,
                    )

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                kv_indices
                            )
                        )

                        kv_indices = self._transform_table_1_to_real(kv_indices)
                        swa_page_table = self._transform_table_1_to_real(swa_page_table)
                    elif self.page_size > 1:
                        kv_indices = self._transform_table_1_to_real(kv_indices)

                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.kv_last_page_len[:bs], dim=0
                    )

            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_q_len, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                run_graph=False,
                swa_page_table=swa_page_table,
            )

        elif forward_batch.forward_mode.is_draft_extend_v2():
            # EAGLE V2: DRAFT_EXTEND_V2 mode - extend draft KV cache with all predicted tokens
            self._ensure_spec_v2_topk_supported()
            if self.use_mla:
                device = forward_batch.seq_lens.device
                num_draft_tokens = self._resolve_v2_num_draft_tokens()
                qo_indptr = self._set_uniform_qo_indptr(bs, num_draft_tokens, device)

                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)

                kv_indices = self._get_kv_indices_scratch(
                    forward_batch.seq_lens_sum, device
                )

                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )

                if _use_mla_ps_kernel:
                    max_seqlen_qo = num_draft_tokens
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    num_draft_tokens,
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )
        elif forward_batch.forward_mode.is_draft_extend():
            # EAGLE V1: DRAFT_EXTEND mode - uses spec_info.accept_length
            if self.use_mla:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )

                if _use_mla_ps_kernel:
                    max_seqlen_qo = max(forward_batch.extend_seq_lens_cpu)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # Non-MLA draft_extend: use triton extend kernel with causal masking
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )
                kv_indices = kv_indices.to(torch.int64)
                draft_max_extend_len = torch.max(spec_info.accept_length).item()

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    draft_max_extend_len,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=None,
                    max_extend_len=draft_max_extend_len,
                )
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                kv_lens = forward_batch.seq_lens + draft_num
                kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                device = forward_batch.seq_lens.device

                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indices = self._get_kv_indices_scratch(
                    kv_lens_sum,
                    device,
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )

                # if self.kv_cache_dtype == fp8_dtype:
                if _use_mla_ps_kernel:
                    max_seqlen_qo = draft_num
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # Non-MLA target_verify: use triton extend kernel with custom mask
                bs = len(forward_batch.req_pool_indices)
                draft_num = spec_info.draft_token_num

                qo_indptr = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=self.device,
                )

                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                kv_indices = torch.empty(
                    kv_indptr[-1], dtype=torch.int64, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )

                custom_mask = spec_info.custom_mask
                seq_mask_len = draft_num * (forward_batch.seq_lens + draft_num)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    draft_num,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=draft_num,
                )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                extend_no_prefix = False
            else:
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if self.use_mla:
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    spec_info=None,
                )

                max_q_len = self.mla_indices_updater_prefill.max_q_len
                qo_indptr = self.mla_indices_updater_prefill.qo_indptr
                kv_indptr = self.mla_indices_updater_prefill.kv_indptr

                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
                fp8_prefill_kv_indices = None

                if _use_fp8_prefill_attn:
                    tile_q = 256
                    qlen_granularity = tile_q // (self.num_head // self.num_kv_head)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_prefill_ps_meta_data_buffer(
                        bs, max_q_len, qlen_granularity
                    )

                    self.make_mla_prefill_ps_meta_data(
                        qo_indptr,
                        kv_indptr,
                        forward_batch.seq_lens,
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        is_causal=True,
                    )

                    total_s = forward_batch.seq_lens_sum
                    fp8_prefill_kv_indices = torch.arange(
                        total_s, device=self.device, dtype=torch.int32
                    )

                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    self.mla_indices_updater_prefill.kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    fp8_prefill_kv_indices=fp8_prefill_kv_indices,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )

                if self.use_sliding_window_kv_pool:
                    swa_page_table = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            self.indices_updater_prefill.kv_indices
                        )
                    )

                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    swa_page_table=swa_page_table,
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_kv_last_page_len = torch.ones(
            max_bs, dtype=torch.int, device=self.device
        )
        if kv_indices_buf is None:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        # if self.use_mla and (_use_mla_ps_kernel or self.kv_cache_dtype == fp8_dtype):
        if self.use_mla and _use_mla_ps_kernel:
            # for persistent mla_decode_fwd
            max_seqlen_qo = (
                1 if self.num_draft_tokens is None else self.num_draft_tokens
            )

            (
                self.work_metadata,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, max_bs)

            # PR 2852 splitter scratch.
            # Enabled whenever the nh=8 bf16 qlen=2 native kernel is usable.
            # At per-step time, `splitter_active` narrows based on max_q_len:
            #   qlen=1 pure decode  -> pad 1->2, single kernel call
            #   qlen=2 target_verify -> native, no splitter needed
            #   qlen=3               -> pad qlen 3->4, then 2 x qlen=2
            #   qlen=4               -> 2 x qlen=2 split
            self._splitter_enabled = bool(
                getattr(self, "_nh8_native_ps", False)
            )
            if self._splitter_enabled:
                max_num_blocks_per_seq = (
                    self.max_context_len + self.page_size - 1
                ) // self.page_size
                self.cg_split_kv_indices_a = torch.zeros(
                    max_bs * max_num_blocks_per_seq,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.cg_split_kv_indptr_a = torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                )
                self.cg_split_qo_indptr = torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                )
                qhd = self._mla_qk_head_dim
                vhd = self._mla_v_head_dim
                self.cg_split_qa = torch.zeros(
                    max_bs * 2, self.num_head, qhd,
                    dtype=torch.bfloat16, device=self.device,
                )
                self.cg_split_qb = torch.zeros(
                    max_bs * 2, self.num_head, qhd,
                    dtype=torch.bfloat16, device=self.device,
                )
                self.cg_split_out_a = torch.zeros(
                    max_bs * 2, self.num_head, vhd,
                    dtype=torch.bfloat16, device=self.device,
                )
                self.cg_split_out_b = torch.zeros(
                    max_bs * 2, self.num_head, vhd,
                    dtype=torch.bfloat16, device=self.device,
                )
            else:
                self.cg_split_kv_indices_a = None
                self.cg_split_kv_indptr_a = None
                self.cg_split_qo_indptr = None
                self.cg_split_qa = None
                self.cg_split_qb = None
                self.cg_split_out_a = None
                self.cg_split_out_b = None

        else:
            self.work_metadata = None
            self.work_indptr = None
            self.work_info_set = None

            self.reduce_indptr = None
            self.reduce_final_map = None
            self.reduce_partial_map = None

        if self.use_sliding_window_kv_pool:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_swa_page_table = torch.zeros(
                (max_bs, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None

        max_kv_len = torch.max(seq_lens).item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:

                if not self.use_triton_unified_attention:
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    kv_indices = self.cuda_graph_kv_indices
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices,
                        seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    max_q_len = 1
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    page_indices = self.req_to_token[req_pool_indices[:bs], :max_kv_len]

                    if self.use_sliding_window_kv_pool:
                        swa_page_indices = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                page_indices
                            )
                        )

                        page_indices = self._transform_table_1_to_real(page_indices)
                        swa_page_indices = self._transform_table_1_to_real(
                            swa_page_indices
                        )

                        new_rows = swa_page_indices.shape[0]
                        new_cols = swa_page_indices.shape[1]

                        kv_indices[:new_rows, :new_cols].copy_(page_indices)
                        swa_page_table = self.cuda_graph_swa_page_table
                        swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                    elif self.page_size > 1:
                        page_indices = self._transform_table_1_to_real(page_indices)
                        new_rows = page_indices.shape[0]
                        new_cols = page_indices.shape[1]
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )

                    kv_indptr = None
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            # PR 2852 splitter for pure-decode (qlen=1) pre-pads 1->2 and
            # uses the nh=8 qlen=2 native kernel.  Populate split_qo_indptr
            # so forward_decode can emit the qlen=2 call under CUDA graph.
            splitter_active = bool(
                getattr(self, "_splitter_enabled", False)
                and _use_mla_ps_kernel
                and max_q_len == 1
                and self.use_mla
                # qlen=1 -> qlen=2 pad: real at position 1, dummy at position 0
            )
            split_qo_indptr = None
            if splitter_active:
                split_qo_indptr = self.cg_split_qo_indptr[: bs + 1]
                split_qo_indptr[: bs + 1] = torch.arange(
                    0, (1 + bs) * 2, step=2,
                    dtype=torch.int32, device=self.device,
                )

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
                splitter_active=splitter_active,
                splitter_qlen=1 if splitter_active else 0,
                split_qo_indptr=split_qo_indptr,
            )

        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                kv_lens = seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                splitter_active = bool(
                    getattr(self, "_splitter_enabled", False)
                    and _use_mla_ps_kernel
                    and max_q_len in (1, 3, 4)
                )
                split_qo_indptr = None
                split_kv_indptr_a = None
                split_kv_indices_a = None
                if splitter_active:
                    (
                        split_qo_indptr,
                        split_kv_indptr_a,
                        split_kv_indices_a,
                    ) = self._populate_splitter_metadata(
                        bs, seq_lens, req_pool_indices
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    splitter_active=splitter_active,
                    splitter_qlen=max_q_len if splitter_active else 0,
                    split_qo_indptr=split_qo_indptr,
                    split_kv_indptr_a=split_kv_indptr_a,
                    split_kv_indices_a=split_kv_indices_a,
                )
            else:
                custom_mask = self.cuda_graph_custom_mask
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Uses fixed num_draft_tokens per batch
            self._ensure_spec_v2_topk_supported()
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            qo_indptr = self._set_uniform_qo_indptr(bs, num_tokens_per_bs, self.device)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses speculative_num_steps + 1
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs

                if _use_mla_ps_kernel:

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                # Non-MLA draft_extend cuda graph: use triton extend kernel
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
                )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None
        max_kv_len = seq_lens_cpu.max().item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:
                if not self.use_triton_unified_attention:
                    kv_indptr = self.kv_indptr
                    kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                    kv_indptr = kv_indptr[: bs + 1]
                    kv_indices = self.cuda_graph_kv_indices
                    create_flashinfer_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        req_pool_indices,
                        seq_lens,
                        kv_indptr,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                    )
                else:
                    max_q_len = 1
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    page_indices = self.req_to_token[req_pool_indices[:bs], :max_kv_len]

                    if self.use_sliding_window_kv_pool:
                        swa_page_indices = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                page_indices
                            )
                        )

                        page_indices = self._transform_table_1_to_real(page_indices)
                        swa_page_indices = self._transform_table_1_to_real(
                            swa_page_indices
                        )

                        new_rows = swa_page_indices.shape[0]
                        new_cols = swa_page_indices.shape[1]

                        kv_indices[:new_rows, :new_cols].copy_(page_indices)
                        swa_page_table = self.cuda_graph_swa_page_table
                        swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                    elif self.page_size > 1:
                        page_indices = self._transform_table_1_to_real(page_indices)
                        new_rows = page_indices.shape[0]
                        new_cols = page_indices.shape[1]
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )

                    kv_indptr = None
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            # PR 2852 splitter for pure-decode (qlen=1): mirror of the capture
            # branch.  Populate split_qo_indptr so forward_decode can replay
            # the qlen=2 kernel call under CUDA graph.
            splitter_active = bool(
                getattr(self, "_splitter_enabled", False)
                and _use_mla_ps_kernel
                and max_q_len == 1
                and self.use_mla
                # qlen=1 -> qlen=2 pad: real at position 1, dummy at position 0
            )
            split_qo_indptr = None
            if splitter_active:
                split_qo_indptr = self.cg_split_qo_indptr[: bs + 1]
                split_qo_indptr[: bs + 1] = torch.arange(
                    0, (1 + bs) * 2, step=2,
                    dtype=torch.int32, device=self.device,
                )

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
                splitter_active=splitter_active,
                splitter_qlen=1 if splitter_active else 0,
                split_qo_indptr=split_qo_indptr,
                # num_kv_splits_indptr=num_kv_splits_indptr,
            )

        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                kv_lens = seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                splitter_active = bool(
                    getattr(self, "_splitter_enabled", False)
                    and _use_mla_ps_kernel
                    and max_q_len in (1, 3, 4)
                )
                split_qo_indptr = None
                split_kv_indptr_a = None
                split_kv_indices_a = None
                if splitter_active:
                    (
                        split_qo_indptr,
                        split_kv_indptr_a,
                        split_kv_indices_a,
                    ) = self._populate_splitter_metadata(
                        bs, seq_lens, req_pool_indices
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    splitter_active=splitter_active,
                    splitter_qlen=max_q_len if splitter_active else 0,
                    split_qo_indptr=split_qo_indptr,
                    split_kv_indptr_a=split_kv_indptr_a,
                    split_kv_indices_a=split_kv_indices_a,
                )
            else:
                custom_mask = self.cuda_graph_custom_mask
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr[: bs + 1]
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Fixed num_draft_tokens per batch
            self._ensure_spec_v2_topk_supported()
            seq_lens = seq_lens[:bs]
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            extend_lens = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=seq_lens.device
            )

            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:

                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses spec_info.accept_length
            num_tokens_per_bs = self.speculative_num_steps + 1
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:

                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )

        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1 if self.num_draft_tokens is None else self.num_draft_tokens

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # AITER verify path does not require post-draft buffer patching currently.
        # This override prevents overlap-plan stream mode from failing with the
        # base class NotImplementedError.
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        self.logits_soft_cap = layer.logit_cap

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.k_scale

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
                # both unified attention and sliding window kv pool are active.
                # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
                # use standard set_kv_buffer, as they lack SWA-specific attributes
                # like full_to_swa_index_mapping.
                if (
                    self.use_triton_unified_attention
                    and self.use_sliding_window_kv_pool
                ):

                    token_to_kv_pool = forward_batch.token_to_kv_pool
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                    launch_reshape_and_cache_flash(
                        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        cache_loc,
                        (
                            slot_mapping_swa.long()
                            if layer.sliding_window_size > 0
                            else None
                        ),
                        k_scale=k_descale,
                        v_scale=v_descale,
                    )
                elif self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, k_descale, v_descale
                    )

        if self.use_mla:
            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                if kv_indices.shape[0] == 0 or extend_no_prefix:
                    if _use_fp8_prefill_attn:
                        output = self.mla_fp8_prefill_attn(
                            q,
                            k,
                            v,
                            layer,
                        )
                    else:
                        output = flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            qo_indptr,
                            max_q_len,
                            max_q_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )
                    return output
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )

                    if self.kv_cache_dtype == fp8_dtype:
                        dtype = q.dtype

                        kvc = kvc.to(dtype)
                        k_pe = k_pe.to(dtype)

                    if (
                        _use_fp8_prefill_attn
                        and layer.kv_b_proj.weight.dtype == torch.uint8
                    ):
                        # MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat
                        # into a single kernel (fused_gemm_afp4wfp4_split_cat) that writes k and v
                        # directly in FP8, avoiding a separate elementwise cast
                        k, v = layer.kv_b_proj(
                            (
                                kvc.squeeze(1),
                                k_pe.expand(-1, layer.tp_k_head_num, -1),
                                qk_nope_head_dim,
                                layer.v_head_dim,
                                fp8_dtype,
                            )
                        )[0]
                    else:
                        kv = layer.kv_b_proj(kvc.contiguous())[0]

                        kv = kv.view(
                            -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                        )
                        k, v = torch.split(
                            kv, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                        )
                        k = torch.cat(
                            [
                                k,
                                torch.broadcast_to(
                                    k_pe,
                                    (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                                ),
                            ],
                            dim=-1,
                        )

                    assert (
                        forward_batch.extend_prefix_lens.shape
                        == forward_batch.extend_seq_lens.shape
                    )

                    if _use_fp8_prefill_attn:
                        return self.mla_fp8_prefill_attn(q, k, v, layer)
                    else:
                        return flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr,
                            max_q_len,
                            max_kv_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )

                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        o = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        o = torch.empty_like(q)

                    mla_prefill_fwd(
                        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    return o
            elif forward_batch.forward_mode.is_target_verify():
                fold_info = _try_fold_mla_q(
                    q,
                    layer.tp_q_head_num,
                    layer.qk_head_dim,
                    self.forward_metadata.max_q_len,
                    self.forward_metadata.qo_indptr,
                    kv_dtype=K_Buffer.dtype,
                )
                if fold_info is not None:
                    o_fold = q.new_empty(
                        (
                            fold_info["bs"],
                            fold_info["fold_nhead"],
                            layer.v_head_dim,
                        ),
                        dtype=self.input_dtype,
                    )
                    mla_decode_fwd(
                        fold_info["q"],
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o_fold,
                        fold_info["qo_indptr"],
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        fold_info["max_q_len"],
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                    )
                    return _unfold_mla_o(
                        o_fold, fold_info, layer.tp_q_head_num, layer.v_head_dim
                    )

                o = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                    dtype=self.input_dtype,
                )

                if getattr(self.forward_metadata, "splitter_active", False):
                    return self._run_mla_splitter(
                        q, K_Buffer, o, layer, k_descale, intra_batch_mode
                    )

                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                _use_native_nh8_q2 = (
                    getattr(self, "_nh8_native_ps", False)
                    and self.forward_metadata.max_q_len == 2
                    and layer.tp_q_head_num == 8
                )
                if layer.tp_q_head_num == 8 and not _use_native_nh8_q2:
                    q_in, o_in = self._pad_mla_q_8_to_16(
                        q, q.shape[0], layer.qk_head_dim, layer.v_head_dim
                    )
                else:
                    q_in, o_in = q, o

                mla_decode_fwd(
                    q_in,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o_in,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    work_meta_data=work_metadata,
                    work_indptr=work_indptr,
                    work_info_set=work_info_set,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    q_scale=k_descale,
                    kv_scale=k_descale,
                    intra_batch_mode=intra_batch_mode,
                    num_kv_splits=num_kv_splits,
                )
                if layer.tp_q_head_num == 8 and not _use_native_nh8_q2:
                    o.copy_(o_in[:, :8, :])
                elif _use_native_nh8_q2:
                    o = o_in
                return o
            elif (
                forward_batch.forward_mode.is_draft_extend()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):

                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                if self.forward_metadata.run_graph is not True:

                    bs, q_pad, q_mask = pad_sequence_with_mask(
                        q.view(q.shape[0], -1),
                        qo_indptr[:-1],
                        forward_batch.extend_seq_lens,
                        self.forward_metadata.max_q_len,
                    )
                    q_pad_v = q_pad.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    fold_info = _try_fold_mla_q(
                        q_pad_v,
                        layer.tp_q_head_num,
                        layer.qk_head_dim,
                        self.forward_metadata.max_q_len,
                        self.forward_metadata.qo_indptr,
                        kv_dtype=K_Buffer.dtype,
                    )
                    if fold_info is not None:
                        o_fold = q.new_empty(
                            (
                                fold_info["bs"],
                                fold_info["fold_nhead"],
                                layer.v_head_dim,
                            ),
                            dtype=self.input_dtype,
                        )
                        mla_decode_fwd(
                            fold_info["q"],
                            K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                            o_fold,
                            fold_info["qo_indptr"],
                            self.forward_metadata.kv_indptr,
                            self.forward_metadata.kv_indices,
                            self.forward_metadata.kv_last_page_len,
                            fold_info["max_q_len"],
                            sm_scale=layer.scaling,
                            logit_cap=layer.logit_cap,
                            q_scale=k_descale,
                            kv_scale=k_descale,
                        )
                        o = _unfold_mla_o(
                            o_fold,
                            fold_info,
                            layer.tp_q_head_num,
                            layer.v_head_dim,
                        )
                        total_valid_q = int(qo_indptr[-1].item())
                        return o[:total_valid_q]

                    o = q.new_empty(
                        (
                            bs * self.forward_metadata.max_q_len,
                            layer.tp_q_head_num,
                            layer.v_head_dim,
                        ),
                        dtype=self.input_dtype,
                    )
                    _use_native_nh8_q2 = (
                        getattr(self, "_nh8_native_ps", False)
                        and self.forward_metadata.max_q_len == 2
                        and layer.tp_q_head_num == 8
                    )
                    if layer.tp_q_head_num == 8 and not _use_native_nh8_q2:
                        q_in, o_in = self._pad_mla_q_8_to_16(
                            q_pad_v, q_pad_v.shape[0], layer.qk_head_dim, layer.v_head_dim
                        )
                    else:
                        q_in, o_in = q_pad_v, o
                    mla_decode_fwd(
                        q_in,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o_in,
                        self.forward_metadata.qo_indptr,
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )
                    if layer.tp_q_head_num == 8 and not _use_native_nh8_q2:
                        o.copy_(o_in[:, :8, :])
                    elif _use_native_nh8_q2:
                        o = o_in

                    # q shape[0] equals qo_indptr[-1]; skip GPU->CPU sync.
                    total_valid_q = q.shape[0]
                    return o[:total_valid_q]
                else:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                        dtype=self.input_dtype,
                    )

                    _use_native_nh8_q2 = (
                        getattr(self, "_nh8_native_ps", False)
                        and self.forward_metadata.max_q_len == 2
                        and layer.tp_q_head_num == 8
                    )
                    if layer.tp_q_head_num == 8 and not _use_native_nh8_q2:
                        q_in, o_in = self._pad_mla_q_8_to_16(
                            q, q.shape[0], layer.qk_head_dim, layer.v_head_dim
                        )
                    else:
                        q_in, o_in = q, o

                    mla_decode_fwd(
                        q_in,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o_in,
                        self.forward_metadata.qo_indptr,
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=k_descale,
                        kv_scale=k_descale,
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )
                    return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                # Use triton extend kernel which supports custom masks and causal masking
                if layer.qk_head_dim != layer.v_head_dim:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)

                self.extend_attention_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.custom_mask,
                    True,  # causal
                    self.forward_metadata.mask_indptr,
                    self.forward_metadata.max_extend_len,
                    1.0,  # k_scale
                    1.0,  # v_scale
                    layer.scaling,
                    logit_cap=layer.logit_cap,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            bs0 = forward_batch.batch_size + 1

            # To keep the mha_batch_prefill_func function parameters
            # declare the necessary parameter and assign None as default value
            q_descale = None

            # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                q = q.to(fp8_dtype)
                q_descale = layer.k_scale if layer.k_scale is not None else self.k_scale

            window_size = (-1, -1)
            page_table = self.forward_metadata.kv_indices

            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                window_size = (layer.sliding_window_size, -1)
                if self.forward_metadata.swa_page_table is not None:
                    page_table = self.forward_metadata.swa_page_table

            o = mha_batch_prefill_func(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                self.qo_indptr[:bs0],
                self.forward_metadata.kv_indptr[:bs0],
                page_table,
                self.forward_metadata.max_q_len,
                self.forward_metadata.max_kv_len,
                causal=True,
                logits_soft_cap=self.logits_soft_cap,
                alibi_slopes=None,
                return_lse=False,
                return_attn_probs=False,
                window_size=window_size,
                sink_ptr=sinks,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.input_dtype,
            )
        else:
            o = torch.empty_like(q, dtype=self.input_dtype)

        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.k_scale

        if save_kv_cache:
            # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
            # both unified attention and sliding window kv pool are active.
            # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
            # use standard set_kv_buffer, as they lack SWA-specific attributes
            # like full_to_swa_index_mapping.
            if self.use_triton_unified_attention and self.use_sliding_window_kv_pool:

                token_to_kv_pool = forward_batch.token_to_kv_pool
                k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
                slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                    slot_mapping_swa.long() if layer.sliding_window_size > 0 else None,
                    k_scale=k_descale,
                    v_scale=v_descale,
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if self.use_mla:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

            work_metadata = self.forward_metadata.work_metadata
            work_indptr = self.forward_metadata.work_indptr
            work_info_set = self.forward_metadata.work_info_set

            reduce_indptr = self.forward_metadata.reduce_indptr
            reduce_final_map = self.forward_metadata.reduce_final_map
            reduce_partial_map = self.forward_metadata.reduce_partial_map

            num_kv_splits = self.forward_metadata.num_kv_splits

            _q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            _o3 = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            # PR 2852 splitter intercept: route qlen=1 pure decode through
            # the nh=8 qlen=2 native kernel (pad 1->2, dummy at pos 1).
            if getattr(self.forward_metadata, "splitter_active", False):
                return self._run_mla_splitter(
                    _q3, k_buffer, _o3, layer, k_descale, intra_batch_mode
                )

            if layer.tp_q_head_num == 8:
                _q_in, _o_in = self._pad_mla_q_8_to_16(
                    _q3, _q3.shape[0], layer.qk_head_dim, layer.v_head_dim
                )
            else:
                _q_in, _o_in = _q3, _o3
            mla_decode_fwd(
                _q_in,
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                _o_in,
                self.forward_metadata.qo_indptr,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=work_metadata,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=k_descale,
                kv_scale=k_descale,
                intra_batch_mode=intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
            if layer.tp_q_head_num == 8:
                _o3.copy_(_o_in[:, :8, :])
        else:
            self.logits_soft_cap = layer.logit_cap

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            if self.use_triton_unified_attention:

                bs = forward_batch.batch_size
                window_size = (-1, -1)
                page_table = self.forward_metadata.kv_indices

                if (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                ):
                    window_size = (layer.sliding_window_size - 1, 0)
                    if self.forward_metadata.swa_page_table is not None:
                        page_table = self.forward_metadata.swa_page_table

                o = torch.empty_like(q, dtype=self.input_dtype)

                max_kv_len = page_table.shape[1] * self.page_size

                unified_attention(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k=k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v=v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    out=o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    cu_seqlens_q=self.forward_metadata.qo_indptr,
                    seqused_k=forward_batch.seq_lens,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=window_size,
                    block_table=page_table,
                    softcap=0,
                    q_descale=None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    sinks=sinks,
                )
            else:
                if self.kv_cache_dtype == fp8_dtype:
                    k_cache = k_cache.to(self.input_dtype)
                    v_cache = v_cache.to(self.input_dtype)

                paged_attention_ragged(
                    o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    self.workspace_buffer,
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_cache.view(-1, 1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
                    self.scale,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.kv_last_page_len,
                    1,
                    self.max_num_partitions,
                    None,
                    "auto",
                    "NHD",
                    self.logits_soft_cap,
                    self.k_scale,
                    self.v_scale,
                    None,
                    _AITER_PARTITION_SIZE_ROCM,
                )

        return o


class AiterIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indices = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):

        kv_start_idx = None
        kv_indptr = self.kv_indptr
        qo_indptr = self.qo_indptr
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        bs = len(req_pool_indices)
        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            # (TODO: Kk) WA - CI test_moe_eval_accuracy_large.py
            # mha_batch_prefill reads 128 data to do computatoin
            # if real data is not long enough then original padding value 0 is used
            # but the 0 location will be made nan (noqa) in cuda graph capture mode
            # this will cause the output tensor value becomes nan
            # WA is to assure that last index of pool not changed
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )

            token_num = kv_indptr[-1]
            kv_indices[token_num:] = kv_indices[0]

            extend_lens = seq_lens - prefix_lens

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indices = kv_indices


class AiterMlaIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indptr = None
        self.kv_indices = None
        self.qo_indptr = None
        self.kv_last_page_len = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        bs = len(req_pool_indices)

        kv_indptr = self.attn_backend.kv_indptr

        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.attn_backend.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    kv_lens,
                    kv_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.max_q_len = max_q_len
        self.max_kv_len = max_kv_len


class AiterMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                AiterAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    topk=topk,
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
            self.page_size,
        )

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
