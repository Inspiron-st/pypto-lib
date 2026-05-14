# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod

import torch

from .kv_cache import KvCacheManager
from .types import (
    DecodeBatch,
    DecodeResult,
    GenerateConfig,
    GenerateResult,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RequestState,
    RuntimeModel,
)


class ModelExecutor(ABC):
    """Backend-neutral interface used by ``LLMEngine`` to execute generation."""

    def __init__(self, kv_cache_manager: KvCacheManager) -> None:
        """Store the KV cache manager shared with the engine."""
        self._kv_cache_manager = kv_cache_manager

    @contextlib.contextmanager
    def session(self):
        """Wrap one generation sequence in executor-specific runtime state."""
        yield

    @property
    def profile_verbose(self) -> bool:
        """Return whether model loading and execution should emit stage timings."""
        return False

    def lookup_embeddings(self, model: RuntimeModel, token_ids: torch.Tensor) -> torch.Tensor:
        """Return embedding rows for ``token_ids`` on the model runtime device."""
        token_ids = token_ids.to(device=model.runtime.device, dtype=torch.long)
        return model.embed_tokens.index_select(0, token_ids.view(-1)).view(
            *token_ids.shape,
            model.config.hidden_size,
        )

    def validate_generate_batch(
        self,
        record: ModelRecord,
        batch_size: int,
        config: GenerateConfig,
    ) -> None:
        """Validate executor-specific limits before KV allocation begins."""
        return None

    def prompt_allocation_length(
        self,
        record: ModelRecord,
        prompt_len: int,
        config: GenerateConfig,
    ) -> int:
        """Return the initial KV allocation size for one prompt."""
        return prompt_len

    def try_generate_batch(
        self,
        record: ModelRecord,
        requests: list[RequestState],
        prefill_batch: PrefillBatch,
        config: GenerateConfig,
    ) -> list[GenerateResult] | None:
        """Optionally handle generation with an executor-specific fast path."""
        return None

    @abstractmethod
    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        """Run prompt prefill and return logits for the next token."""
        raise NotImplementedError

    @abstractmethod
    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        """Run one decode step for active requests and return next-token logits."""
        raise NotImplementedError
