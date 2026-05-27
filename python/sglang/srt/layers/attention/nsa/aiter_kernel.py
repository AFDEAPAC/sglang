from typing import Optional, Tuple

import torch

from aiter import dtypes
from aiter.ops.quant import per_group_quant_hip


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-wise FP8 quantization using aiter kernel.
    Outputs torch.float8_e4m3fnuz, which is the correct format for ROCm/MI300X.
    scale_fmt (e.g. "ue8m0") is intentionally ignored; GLM model passes None.
    """
    return per_group_quant_hip(x, quant_dtype=dtypes.fp8, group_size=block_size)
