# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Thin LLM runtime wrapper around :mod:`simpler.worker`.

This module is the boundary between the LLM package and the Simpler runtime.
It keeps model code from constructing Simpler workers directly while still
exposing the primitives the LLM runtime needs: L2/L3 worker lifecycle,
callable dispatch, explicit worker-child memory operations, and tensor handles
that can be submitted as ``child_memory`` inputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from simpler.task_interface import ContinuousTensor, DataType
from simpler.worker import Worker as SimplerWorker


_TORCH_TO_SIMPLER_DTYPE = {
    torch.float32: DataType.FLOAT32,
    torch.float16: DataType.FLOAT16,
    torch.bfloat16: DataType.BFLOAT16,
    torch.int8: DataType.INT8,
    torch.int16: DataType.INT16,
    torch.int32: DataType.INT32,
    torch.int64: DataType.INT64,
    torch.uint8: DataType.UINT8,
    torch.uint16: DataType.UINT16,
    torch.uint32: DataType.UINT32,
}
_SIMPLER_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_TO_SIMPLER_DTYPE.items()}


def _to_simpler_dtype(dtype: torch.dtype | DataType) -> DataType:
    if isinstance(dtype, DataType):
        return dtype
    try:
        return _TORCH_TO_SIMPLER_DTYPE[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported dtype for worker tensor: {dtype!r}") from exc


def _to_torch_dtype(dtype: torch.dtype | DataType) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _SIMPLER_TO_TORCH_DTYPE[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported Simpler dtype for torch conversion: {dtype!r}") from exc


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    shape_t = tuple(shape)
    if not shape_t:
        raise ValueError("shape must be non-empty")
    for dim in shape_t:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(f"shape must contain ints, got {shape_t!r}")
    if any(dim <= 0 for dim in shape_t):
        raise ValueError(f"shape must contain only positive dimensions, got {shape_t}")
    return shape_t


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    n = torch.empty((), dtype=dtype).element_size()
    for dim in shape:
        n *= dim
    return n


@dataclass(frozen=True)
class WorkerTensor:
    """Handle to memory allocated inside a Simpler worker child.

    ``WorkerTensor`` is metadata only.  It does not own the device allocation;
    callers must free it with :meth:`Worker.free_tensor` or
    :meth:`Worker.free`.  Use :meth:`to_continuous_tensor` when submitting a
    kernel argument that should be treated as pre-existing child memory.
    """

    data_ptr: int
    shape: tuple[int, ...]
    dtype: DataType
    worker_id: int = 0

    def __post_init__(self) -> None:
        """Validate that the handle describes a real worker allocation."""
        _validate_positive_int("data_ptr", self.data_ptr)
        object.__setattr__(self, "shape", _normalize_shape(self.shape))
        if not isinstance(self.dtype, DataType):
            raise TypeError(f"dtype must be simpler.task_interface.DataType, got {type(self.dtype).__name__}")
        if isinstance(self.worker_id, bool) or not isinstance(self.worker_id, int) or self.worker_id < 0:
            raise ValueError(f"worker_id must be a non-negative int, got {self.worker_id!r}")

    @property
    def nbytes(self) -> int:
        """Number of bytes covered by the logical tensor view."""
        return _nbytes(self.shape, self.torch_dtype)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the corresponding ``torch.dtype``."""
        return _to_torch_dtype(self.dtype)

    def to_continuous_tensor(self) -> ContinuousTensor:
        """Return a Simpler tensor view that skips runtime malloc/free."""
        return ContinuousTensor.make(self.data_ptr, self.shape, self.dtype, child_memory=True)


class Worker:
    """Manage a Simpler L2 or L3 worker for LLM execution.

    ``Worker`` owns the underlying ``simpler.worker.Worker`` instance and
    forwards the runtime operations used by model runners.  Level 2 targets a
    single chip; level 3 targets a host-level worker with chip children and
    optional Python sub-workers.  The wrapper is intentionally small: it adds
    LLM-oriented validation and ``WorkerTensor`` helpers, but preserves Simpler
    concepts such as callables, worker ids, ``CallConfig``, and
    ``ContinuousTensor``.
    """

    def __init__(
        self,
        *,
        level: int = 2,
        platform: str = "a2a3",
        runtime: str = "tensormap_and_ringbuffer",
        device_id: int = 0,
        device_ids: Sequence[int] | None = None,
        num_sub_workers: int = 0,
        auto_init: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a Simpler-backed L2 or L3 worker.

        ``level=2`` binds to one chip via ``device_id``.  ``level=3`` creates
        a hierarchical worker with chip children from ``device_ids`` plus
        optional Python sub-workers.  Extra keyword arguments are forwarded to
        ``simpler.worker.Worker`` so runtime-specific knobs can pass through
        without growing this wrapper.
        """
        if level not in (2, 3):
            raise ValueError(f"llm.runtime.Worker supports only level 2 or 3, got {level}")

        self.level = int(level)
        self.platform = platform
        self.runtime = runtime
        self.device_id = int(device_id)
        self.device_ids = [int(d) for d in (device_ids if device_ids is not None else [device_id])]
        self.num_sub_workers = int(num_sub_workers)
        self._initialized = False

        if self.level == 2:
            self._worker = SimplerWorker(
                level=2,
                platform=platform,
                runtime=runtime,
                device_id=self.device_id,
                **kwargs,
            )
        else:
            self._worker = SimplerWorker(
                level=3,
                platform=platform,
                runtime=runtime,
                device_ids=self.device_ids,
                num_sub_workers=self.num_sub_workers,
                **kwargs,
            )

        if auto_init:
            self.init()

    @property
    def impl(self) -> SimplerWorker:
        """Return the wrapped Simpler worker for advanced runtime code."""
        return self._worker

    @property
    def initialized(self) -> bool:
        """Whether the wrapped Simpler worker has been initialized."""
        return self._initialized

    def init(self) -> None:
        """Initialize runtime resources if they are not already live."""
        if self._initialized:
            return
        self._worker.init()
        self._initialized = True

    def close(self) -> None:
        """Release runtime resources if initialized."""
        if not self._initialized:
            return
        self._worker.close()
        self._initialized = False

    def _require_initialized(self, op: str) -> None:
        if not self._initialized:
            raise RuntimeError(f"Worker.{op} requires init() first")

    def register(self, callable_obj: Any) -> int:
        """Register a chip callable, orchestration function, or sub-worker function."""
        return self._worker.register(callable_obj)

    def unregister_callable(self, callable_id: int) -> None:
        """Unregister a prepared L2 callable id."""
        self._require_initialized("unregister_callable")
        self._worker.unregister_callable(int(callable_id))

    def prepare_callable(self, callable_id: int, callable_obj: Any) -> None:
        """Pre-stage an L2 chip callable under ``callable_id``."""
        self._require_initialized("prepare_callable")
        self._worker.prepare_callable(int(callable_id), callable_obj)

    def run(self, callable_obj: Any, args: Any = None, config: Any = None) -> None:
        """Run one L2 callable id or one L3 orchestration function."""
        self._require_initialized("run")
        self._worker.run(callable_obj, args=args, config=config)

    def run_prepared(self, callable_id: int, args: Any = None, config: Any = None) -> None:
        """Run a callable previously staged with :meth:`prepare_callable`."""
        self._require_initialized("run_prepared")
        self._worker.run_prepared(int(callable_id), args=args, config=config)

    def malloc(self, nbytes: int, *, worker_id: int = 0) -> int:
        """Allocate bytes on a worker child and return the device pointer."""
        self._require_initialized("malloc")
        nbytes = _validate_positive_int("nbytes", nbytes)
        return self._worker.malloc(nbytes, worker_id=int(worker_id))

    def free(self, ptr: int, *, worker_id: int = 0) -> None:
        """Free a pointer previously returned by :meth:`malloc`."""
        self._require_initialized("free")
        self._worker.free(int(ptr), worker_id=int(worker_id))

    def copy_to(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        """Copy ``nbytes`` from a host pointer to worker device memory."""
        self._require_initialized("copy_to")
        self._worker.copy_to(int(dst), int(src), _validate_positive_int("nbytes", nbytes), worker_id=int(worker_id))

    def copy_from(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        """Copy ``nbytes`` from worker device memory to a host pointer."""
        self._require_initialized("copy_from")
        self._worker.copy_from(int(dst), int(src), _validate_positive_int("nbytes", nbytes), worker_id=int(worker_id))

    def alloc_tensor(
        self,
        shape: Sequence[int],
        dtype: torch.dtype | DataType,
        *,
        init: torch.Tensor | None = None,
        worker_id: int = 0,
    ) -> WorkerTensor:
        """Allocate a worker-resident tensor and optionally upload host data.

        The returned ``WorkerTensor`` is suitable for LLM weights or KV-cache
        buffers that should stay resident across submitted tasks.  If
        initialization fails after allocation, the device pointer is freed
        before the exception is re-raised.
        """
        shape_t = _normalize_shape(shape)

        simpler_dtype = _to_simpler_dtype(dtype)
        torch_dtype = _to_torch_dtype(dtype)
        nbytes = _nbytes(shape_t, torch_dtype)

        ptr = self.malloc(nbytes, worker_id=worker_id)
        try:
            if init is not None:
                if tuple(init.shape) != shape_t or init.dtype != torch_dtype:
                    raise ValueError(
                        f"init must have shape={shape_t} dtype={torch_dtype}, "
                        f"got shape={tuple(init.shape)} dtype={init.dtype}"
                    )
                host = init.contiguous().cpu()
                self.copy_to(ptr, host.data_ptr(), nbytes, worker_id=worker_id)
            return WorkerTensor(ptr, shape_t, simpler_dtype, int(worker_id))
        except Exception:
            self.free(ptr, worker_id=worker_id)
            raise

    def free_tensor(self, tensor: WorkerTensor) -> None:
        """Free a tensor handle returned by :meth:`alloc_tensor`."""
        self.free(tensor.data_ptr, worker_id=tensor.worker_id)

    def __enter__(self) -> "Worker":
        """Initialize and return this worker for ``with`` blocks."""
        self.init()
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Close this worker when leaving a ``with`` block."""
        self.close()


__all__ = ["Worker", "WorkerTensor"]
