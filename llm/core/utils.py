# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import torch


def backend_type_for_platform(platform: str):
    """Map a platform name to the PyPTO backend enum used by runtime config."""
    from pypto.backend import BackendType

    if platform.startswith("a5"):
        return BackendType.Ascend950
    return BackendType.Ascend910B


def rope_tables(max_seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine RoPE lookup tables for the configured context."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    return torch.cat([cos_half, cos_half], dim=-1), torch.cat([sin_half, sin_half], dim=-1)


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple
