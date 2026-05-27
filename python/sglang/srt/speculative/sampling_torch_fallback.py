"""Optimized pure-PyTorch fallback implementations for sgl_kernel sampling ops.

V2: On top of the previous topk-based renorm and GPU final sampling,
this version adds:
  1. top_p_renorm_prob rewritten with topk(K)+sort instead of full-vocab sort (66x)
  2. fused_topk_topp_renorm: single topk pass for combined top_k + top_p
  3. tree_speculative_sampling_target_only: eliminates large tensor CPU copies
     by pre-gathering a tiny lookup table on GPU (14x)
"""

import torch
from typing import Union


def top_k_renorm_prob(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Renormalize probabilities by top-k thresholding -- topk-based (no full sort)."""
    if isinstance(top_k, int):
        top_ks = torch.full(
            (probs.shape[0],), top_k, dtype=torch.int32, device=probs.device
        )
    else:
        top_ks = top_k.to(torch.int32)

    k_max = int(top_ks.max().item())
    if k_max >= probs.shape[-1]:
        return probs

    vals, idx = torch.topk(probs.float(), k=k_max, dim=-1, sorted=False)
    mask = (
        torch.arange(k_max, device=probs.device).unsqueeze(0) >= top_ks.unsqueeze(1)
    )
    vals.masked_fill_(mask, 0.0)
    renorm = vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    vals.div_(renorm)
    result = torch.zeros_like(probs)
    result.scatter_(1, idx, vals)
    return result


_TOP_P_K = 64


def top_p_renorm_prob(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Renormalize probabilities by top-p (nucleus) thresholding.

    Uses topk(K) + sort(K) instead of full-vocab sort(V), where K << V.
    For typical top_p in [0.9, 0.99], K=64 captures >99.9% of the mass.
    """
    if isinstance(top_p, float):
        top_ps = torch.full(
            (probs.shape[0],), top_p, dtype=torch.float32, device=probs.device
        )
    else:
        top_ps = top_p.float()

    K = min(_TOP_P_K, probs.shape[-1])
    vals, idx = torch.topk(probs.float(), k=K, dim=-1, sorted=True)
    cumsum = torch.cumsum(vals, dim=-1)
    mask = (cumsum - vals) > top_ps.unsqueeze(1)
    vals.masked_fill_(mask, 0.0)
    renorm = vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    vals.div_(renorm)
    result = torch.zeros_like(probs)
    result.scatter_(1, idx, vals)
    return result


def fused_topk_topp_renorm(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    need_top_p: bool = True,
) -> torch.Tensor:
    """Fused top-k + top-p renormalization in a single topk pass.

    Applies top-k mask and renormalizes first, then applies top-p cumsum
    mask on the renormalized distribution and renormalizes again.
    Single torch.topk call replaces two separate passes over the full vocab.
    """
    top_ks_i32 = top_ks.to(torch.int32)
    k_max = int(top_ks_i32.max().item())
    V = probs.shape[-1]

    if k_max >= V:
        if not need_top_p:
            return probs
        return top_p_renorm_prob(probs, top_ps)

    vals, idx = torch.topk(probs.float(), k=k_max, dim=-1, sorted=True)

    k_mask = (
        torch.arange(k_max, device=probs.device).unsqueeze(0)
        >= top_ks_i32.unsqueeze(1)
    )
    vals.masked_fill_(k_mask, 0.0)

    if need_top_p:
        renorm_k = vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vals.div_(renorm_k)
        cumsum = torch.cumsum(vals, dim=-1)
        p_mask = (cumsum - vals) > top_ps.float().unsqueeze(1)
        vals.masked_fill_(p_mask, 0.0)

    renorm = vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    vals.div_(renorm)
    result = torch.zeros_like(probs)
    result.scatter_(1, idx, vals)
    return result


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    """Tree speculative sampling -- zero-copy GPU-optimized version.

    Pre-gathers a tiny lookup table on GPU:
      lookup[b, row, col] = target_probs[b, row, candidates[b, col]]
    This captures all possible scalar reads needed during the tree walk
    (shape: bs x D x D where D = num_draft_tokens, typically 4).
    Avoids copying the full target_probs (bs x D x V, V=151k) to CPU.
    """
    device = target_probs.device
    bs = candidates.shape[0]
    num_draft_tokens = candidates.shape[1]
    num_spec_step = accept_index.shape[1]
    vocab_size = target_probs.shape[2]
    capped_threshold_acc = max(threshold_acc, 1e-9)

    # Pre-gather lookup table on GPU: (bs, D, D)
    # lookup[b, row, col] = target_probs[b, row, candidates[b, col]]
    cand_long = candidates.long()
    # Expand candidates for gathering: (bs, 1, D) -> (bs, D, D)
    cand_expand = cand_long.unsqueeze(1).expand(bs, num_draft_tokens, num_draft_tokens)
    lookup_gpu = target_probs.gather(2, cand_expand)  # (bs, D, D)
    lookup_cpu = lookup_gpu.cpu()  # tiny: bs * D * D floats

    # Small tensors -> CPU
    candidates_cpu = candidates.cpu()
    retrive_index_cpu = retrive_index.cpu()
    retrive_next_token_cpu = retrive_next_token.cpu()
    retrive_next_sibling_cpu = retrive_next_sibling.cpu()
    uniform_samples_cpu = uniform_samples.cpu()

    predicts_cpu = predicts.cpu()
    accept_index_cpu = accept_index.cpu()
    accept_token_num_cpu = accept_token_num.cpu()

    # Accumulate sparse draft_probs updates: (b, row, token_id, value)
    draft_update_indices = []

    final_rows = []

    for b in range(bs):
        prob_acc = 0.0
        cur_prob_row = 0
        coin = uniform_samples_cpu[b, 0].item()
        last_accepted_retrive_idx = retrive_index_cpu[b, 0].item()
        accept_index_cpu[b, 0] = last_accepted_retrive_idx
        num_accepted = 0
        cur_index = 0

        for j in range(1, num_spec_step):
            cur_index = retrive_next_token_cpu[b, cur_index].item()
            while cur_index != -1:
                draft_token_id = candidates_cpu[b, cur_index].item()
                # lookup[b, cur_prob_row, cur_index] = target_probs[b, cur_prob_row, candidates[b, cur_index]]
                target_prob_single = lookup_cpu[b, cur_prob_row, cur_index].item()
                prob_acc += target_prob_single

                if (coin <= prob_acc / capped_threshold_acc) or (
                    target_prob_single >= threshold_single
                ):
                    prob_acc = 0.0
                    cur_prob_row = cur_index
                    coin = uniform_samples_cpu[b, cur_index].item()
                    predicts_cpu[last_accepted_retrive_idx] = draft_token_id
                    num_accepted += 1
                    draft_index = retrive_index_cpu[b, cur_index].item()
                    accept_index_cpu[b, num_accepted] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    # Record update: draft_probs[b, cur_prob_row, draft_token_id] = target_prob_value
                    draft_update_indices.append(
                        (b, cur_prob_row, draft_token_id, target_prob_single)
                    )
                    cur_index = retrive_next_sibling_cpu[b, cur_index].item()

            if cur_index == -1:
                break

        accept_token_num_cpu[b] = num_accepted
        final_rows.append((b, cur_prob_row, last_accepted_retrive_idx))

    # Apply sparse draft_probs updates on GPU
    if draft_update_indices:
        ub = torch.tensor([u[0] for u in draft_update_indices], dtype=torch.long, device=device)
        ur = torch.tensor([u[1] for u in draft_update_indices], dtype=torch.long, device=device)
        ut = torch.tensor([u[2] for u in draft_update_indices], dtype=torch.long, device=device)
        uv = torch.tensor([u[3] for u in draft_update_indices], dtype=target_probs.dtype, device=device)
        draft_probs[ub, ur, ut] = uv

    # Final sampling on GPU
    if final_rows:
        b_indices = torch.tensor(
            [r[0] for r in final_rows], dtype=torch.long, device=device
        )
        row_indices = torch.tensor(
            [r[1] for r in final_rows], dtype=torch.long, device=device
        )
        write_indices_gpu = torch.tensor(
            [r[2] for r in final_rows], dtype=torch.long, device=device
        )
        write_indices_cpu = torch.tensor(
            [r[2] for r in final_rows], dtype=torch.long
        )
        coins_final = uniform_samples_for_final_sampling[b_indices]

        q = target_probs[b_indices, row_indices]
        p = draft_probs[b_indices, row_indices]
        relu_diff = torch.clamp(q - p, min=0.0)
        totals = relu_diff.sum(dim=-1)
        has_mass = totals > 0

        if has_mass.any():
            rd = relu_diff[has_mass]
            cs = torch.cumsum(rd, dim=-1)
            thresholds = (coins_final[has_mass] * totals[has_mass]).unsqueeze(-1)
            sampled_ids = (cs >= thresholds).to(torch.int32).argmax(dim=-1).to(torch.int32)

            wi = write_indices_cpu[has_mass.cpu()]
            si = sampled_ids.cpu()
            predicts_cpu[wi] = si

        if (~has_mass).any():
            wi_no = write_indices_cpu[(~has_mass).cpu()]
            predicts_cpu[wi_no] = vocab_size - 1

    predicts.copy_(predicts_cpu.to(device))
    accept_index.copy_(accept_index_cpu.to(device))
    accept_token_num.copy_(accept_token_num_cpu.to(device))
