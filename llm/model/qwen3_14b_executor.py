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


try:
    from ..core._profiling import StageTimer
    from ..core.kv_cache import KvCacheManager
    from ..core.model_runner import ModelRunner
    from ..core.pypto_executor import PyptoExecutor as CorePyptoExecutor
    from ..core.types import (
        GenerateConfig,
        GenerateResult,
        ModelRecord,
        PrefillBatch,
        RequestState,
        RuntimeModel,
    )
    from ..core.utils import rope_tables, round_up
    from .qwen3_14b_runner import (
        _CompiledKernels,
        _KernelLayerWeights,
        _LOGITS_BATCH_TILE,
        Qwen314BModelRunner,
    )
except ImportError:
    from core._profiling import StageTimer
    from core.kv_cache import KvCacheManager
    from core.model_runner import ModelRunner
    from core.pypto_executor import PyptoExecutor as CorePyptoExecutor
    from core.types import (
        GenerateConfig,
        GenerateResult,
        ModelRecord,
        PrefillBatch,
        RequestState,
        RuntimeModel,
    )
    from core.utils import rope_tables, round_up
    from model.qwen3_14b_runner import (
        _CompiledKernels,
        _KernelLayerWeights,
        _LOGITS_BATCH_TILE,
        Qwen314BModelRunner,
    )


_VOCAB_PAD_MULTIPLE = 512  # must be a multiple of qwen3_14b_lm_head.VOCAB_CHUNK (64)
_QWEN14B_PAGE_SIZE = 256


def _patch_orch_make_tensor_arg(module: object) -> None:
    """Allow ``make_tensor_arg`` in a generated orchestration module to pass
    ``ContinuousTensor`` objects through unchanged.

    The generated ``host_orch.py`` always calls ``make_tensor_arg(t)`` for
    every entry in the ``tensors`` dict.  When we pre-upload static weights
    and replace them with ``ContinuousTensor(child_memory=True)`` objects,
    the default ``make_tensor_arg`` (which expects a ``torch.Tensor``) would
    crash.  Patching it here lets child_memory tensors pass through as-is
    so the runtime skips H2D/D2H for those buffers.
    """
    try:
        from simpler.task_interface import ContinuousTensor  # noqa: PLC0415
    except ImportError:
        return
    _orig = getattr(module, "make_tensor_arg", None)
    if _orig is None or getattr(_orig, "_child_memory_patched", False):
        return

    def _patched(tensor: object) -> object:
        if isinstance(tensor, ContinuousTensor):
            return tensor
        return _orig(tensor)  # type: ignore[misc]

    _patched._child_memory_patched = True  # type: ignore[attr-defined]
    module.make_tensor_arg = _patched  # type: ignore[attr-defined]


class _StackedLayerView:
    """Adapter exposing HF-format LayerWeights in kernel orientation.

    stack_layer_weights() expects each per-layer weight already in the
    orientation the kernel ingests (transposed BF16 contiguous CPU). This
    view computes that view lazily per attribute access so the stacker can
    iterate ``getattr(layer, attr)`` against the standard LayerWeights.
    """

    _KERNEL_2D_ATTRS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")

    def __init__(self, layer) -> None:
        self._layer = layer

    def __getattr__(self, name: str) -> torch.Tensor:
        weight = getattr(self._layer, name)
        if name in self._KERNEL_2D_ATTRS:
            return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu()
        # Norm gammas (input_rms_weight, q/k/post norm). The stacker
        # flattens to [dim] then re-stacks; cast to FP32 to match the
        # kernel signature ([num_layers, dim], FP32).
        return weight.view(-1).float().contiguous().cpu()


class Qwen314BPyptoExecutor(CorePyptoExecutor):
    """PyPTO executor that compiles and registers the Qwen3-14B kernels."""

    def __init__(
        self,
        kv_cache_manager: KvCacheManager,
        *,
        platform: str = "a2a3sim",
        device_id: int = 0,
        save_kernels_dir: str | None = None,
        l3_mode: bool = False,
        l3_trace: bool = False,
    ) -> None:
        super().__init__(
            kv_cache_manager,
            platform=platform,
            device_id=device_id,
            save_kernels_dir=save_kernels_dir,
        )
        self._l3_mode = l3_mode
        self._l3_trace = l3_trace

    @property
    def profile_verbose(self) -> bool:
        """Return whether compile and L3 execution timing logs are enabled."""
        return self._l3_trace

    def _create_runner(self, model_id: str, compiled: object) -> ModelRunner:
        """Create the Qwen3-14B runtime runner for compiled kernels."""
        if not isinstance(compiled, _CompiledKernels):
            raise TypeError("Qwen314BPyptoExecutor requires Qwen3-14B compiled kernels.")
        return Qwen314BModelRunner(
            model_id=model_id,
            compiled=compiled,
            kv_cache_manager=self._kv_cache_manager,
            platform=self._platform,
            device_id=self._device_id,
            save_kernels_dir=self._save_kernels_dir,
            l3_trace=self._l3_trace,
        )

    def validate_generate_batch(
        self,
        record: ModelRecord,
        batch_size: int,
        config: GenerateConfig,
    ) -> None:
        """Reject generation settings unsupported by the Qwen3-14B L3 path."""
        if not self._l3_mode:
            return
        if batch_size != 1:
            raise NotImplementedError(
                "L3 generate fast path currently supports batch_size=1; "
                f"got {batch_size} prompts."
            )
        if any(config.stop):
            raise NotImplementedError(
                "L3 generate fast path does not support generate_config.stop; "
                "the device-side decode loop cannot break on host-side string matches."
            )

    def prompt_allocation_length(
        self,
        record: ModelRecord,
        prompt_len: int,
        config: GenerateConfig,
    ) -> int:
        """Return prompt KV length, including generated-token capacity for L3."""
        if not self._l3_mode:
            return prompt_len
        max_seq = record.runtime.max_seq_len
        if config.max_new_tokens < 1:
            raise ValueError(
                f"L3 mode requires max_new_tokens >= 1, got {config.max_new_tokens}."
            )
        if prompt_len > max_seq:
            raise ValueError(
                f"L3 mode: prompt length ({prompt_len}) exceeds max_seq_len ({max_seq}). "
                f"Either shorten the prompt or increase --max-seq-len to at least {prompt_len}."
            )
        alloc_len = prompt_len + config.max_new_tokens
        if alloc_len > max_seq:
            allowed_new_tokens = max_seq - prompt_len
            raise ValueError(
                f"L3 mode: prompt length ({prompt_len}) + max_new_tokens "
                f"({config.max_new_tokens}) = {alloc_len} exceeds max_seq_len ({max_seq}). "
                f"Either reduce --max-new-tokens to at most {allowed_new_tokens}, "
                f"or increase --max-seq-len to at least {alloc_len}."
            )
        return alloc_len

    def try_generate_batch(
        self,
        record: ModelRecord,
        requests: list[RequestState],
        prefill_batch: PrefillBatch,
        config: GenerateConfig,
    ) -> list[GenerateResult] | None:
        """Run the single-dispatch L3 generation path when L3 mode is enabled."""
        if not self._l3_mode:
            return None
        generated_ids, _ = self.run_generate_l3(
            record.runtime_model,
            prefill_batch,
            max_new_tokens=config.max_new_tokens,
            eos_token_id=record.config.eos_token_id,
        )
        request = requests[0]
        request.generated_token_ids = list(generated_ids)
        request.output_text = record.tokenizer.decode(generated_ids)
        if (
            generated_ids
            and record.config.eos_token_id is not None
            and generated_ids[-1] == record.config.eos_token_id
        ):
            finish_reason = "eos"
        else:
            finish_reason = "length"
        return [
            GenerateResult(
                text=request.output_text,
                token_ids=list(request.generated_token_ids),
                finish_reason=finish_reason,
            )
        ]

    def run_generate_l3(
        self,
        model: RuntimeModel,
        prefill_batch: PrefillBatch,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> tuple[list[int], torch.Tensor]:
        """Delegate L3 generation to the registered Qwen3-14B runner."""
        runner = self._runners[model.config.model_id]
        if not isinstance(runner, Qwen314BModelRunner):
            raise TypeError("Qwen314BPyptoExecutor requires a Qwen314BModelRunner.")
        return runner.run_generate_l3(
            model,
            prefill_batch,
            max_new_tokens,
            eos_token_id,
        )

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        """Compile Qwen3-14B PyPTO kernels and pack runtime artifacts."""
        timer = StageTimer(
            enabled=self._l3_trace,
            prefix="compile-breakdown",
            title="_compile_model stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        from pypto.runtime import run
        try:
            from ..model.qwen3_14b_decode import build_qwen3_decode_program
            from ..model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from ..model.qwen3_14b_l3_generate import (
                build_qwen3_14b_l3_generate_program,
                stack_layer_weights_full,
            )
            from ..model.qwen3_14b_lm_head import build_qwen3_lm_head_program
            from ..model.qwen3_14b_prefill import build_qwen3_14b_prefill_program
        except ImportError:
            from model.qwen3_14b_decode import build_qwen3_decode_program
            from model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from model.qwen3_14b_l3_generate import (
                build_qwen3_14b_l3_generate_program,
                stack_layer_weights_full,
            )
            from model.qwen3_14b_lm_head import build_qwen3_lm_head_program
            from model.qwen3_14b_prefill import build_qwen3_14b_prefill_program
        _mark("imports")

        self._validate_supported_shape(model)
        kernel_batch = model.runtime.max_batch_size
        self._validate_total_kv_pages(model, kernel_batch)

        prefill_program = build_qwen3_14b_prefill_program(
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            intermediate_size=model.config.intermediate_size,
        )
        decode_program = build_qwen3_decode_program(
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
        )
        padded_vocab = round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        final_rms_program = build_qwen3_final_rms_program(
            batch=_LOGITS_BATCH_TILE,
            hidden_size=model.config.hidden_size,
            eps=model.config.rms_norm_eps,
        )
        lm_head_program = build_qwen3_lm_head_program(
            batch=_LOGITS_BATCH_TILE,
            hidden_size=model.config.hidden_size,
            vocab_size=padded_vocab,
        )
        _mark("build_programs")
        prefill = run(prefill_program, config=self._run_config(codegen_only=True))
        _mark("compile_prefill")
        decode = run(decode_program, config=self._run_config(codegen_only=True))
        _mark("compile_decode")
        final_rms = run(final_rms_program, config=self._run_config(codegen_only=True))
        lm_head = run(lm_head_program, config=self._run_config(codegen_only=True))
        _mark("compile_final_rms_lm_head")
        rope_cos, rope_sin = rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )

        _mark("rope_tables")

        # L3 artefacts (built only when l3_mode=True).
        l3_generate: object | None = None
        stacked_weights: dict[str, torch.Tensor] | None = None
        if self._l3_mode:
            l3_generate_program = build_qwen3_14b_l3_generate_program(
                num_layers=model.config.num_hidden_layers,
                batch=kernel_batch,
                max_seq=model.runtime.max_seq_len,
                hidden_size=model.config.hidden_size,
                intermediate_size=model.config.intermediate_size,
                num_heads=model.config.num_attention_heads,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
                max_new_tokens=model.runtime.max_new_tokens,
                padded_vocab=padded_vocab,
                page_size=model.runtime.page_size,
            )
            # pypto.runtime.run hard-codes DistributedConfig(block_dim=1), so we call
            # ir.compile directly to override it.
            from pypto import ir  # noqa: PLC0415
            from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

            _rc = self._run_config(codegen_only=True)
            l3_generate = ir.compile(
                l3_generate_program,
                output_dir=_rc.save_kernels_dir,
                strategy=_rc.strategy,
                backend_type=_rc.backend_type,
                dump_passes=_rc.dump_passes,
                diagnostic_phase=_rc.diagnostic_phase,
                disabled_diagnostics=_rc.disabled_diagnostics,
                platform=_rc.platform,
                profiling=_rc.compile_profiling,
                distributed_config=DistributedConfig(
                    device_ids=[self._device_id],
                    block_dim=3,
                    num_sub_workers=0,
                    aicpu_thread_num=4,
                ),
            )
            _mark("compile_l3_generate")
            # stack_layer_weights_full expects each weight already in
            # kernel orientation ([in_dim, out_dim] for 2D matmul weights).
            # Adapt via a per-layer view that mirrors _kernel_weight().
            stacked_layers = [_StackedLayerView(layer) for layer in model.layers]
            stacked_weights = stack_layer_weights_full(
                stacked_layers,
                hidden=model.config.hidden_size,
                kv_hidden=model.config.num_key_value_heads * model.config.head_dim,
                inter=model.config.intermediate_size,
                head_dim=model.config.head_dim,
            )
            _mark("stack_layer_weights")

        # L3-wrapped generate: pre-extract l3_generate setup artifacts
        # (expensive compile_and_assemble + module loading done once,
        # reused per generate call).
        l3_generate_chip_callables: dict[str, object] | None = None
        l3_generate_entry_fn: object | None = None
        l3_generate_sub_worker_fns: dict[str, object] | None = None
        l3_generate_dc: object | None = None
        l3_generate_platform: str | None = None
        l3_generate_runtime_name: str | None = None
        l3_generate_param_infos: object | None = None
        if self._l3_mode:
            from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415
            from pypto.runtime.distributed_runner import _load_generated_module  # noqa: PLC0415
            from pypto.pypto_core.ir import FunctionType  # noqa: PLC0415

            # Pre-extract l3_generate setup artifacts.
            lg_dc = l3_generate._distributed_config
            lg_output_dir = l3_generate.output_dir
            lg_chip_callables: dict[str, object] = {}
            lg_runtime_name = "tensormap_and_ringbuffer"
            lg_next_levels_dir = lg_output_dir / "next_levels"
            for func in l3_generate._program.functions.values():
                if func.func_type in (FunctionType.Orchestration, FunctionType.Opaque):
                    chip_dir = lg_next_levels_dir / func.name
                    if chip_dir.exists():
                        cc, lg_runtime_name = compile_and_assemble(
                            chip_dir, l3_generate.platform,
                        )
                        lg_chip_callables[func.name] = cc
            lg_orch_path = lg_output_dir / "orchestration" / "host_orch.py"
            lg_orch_module = _load_generated_module(lg_orch_path)
            # Patch make_tensor_arg so pre-uploaded ContinuousTensor objects
            # (child_memory=True) are passed through unchanged instead of
            # triggering a crash when the generated code calls make_tensor_arg
            # on a non-torch.Tensor value.
            _patch_orch_make_tensor_arg(lg_orch_module)
            lg_entry_fn = None
            for attr_name in ("entry", "host_orch"):
                lg_entry_fn = getattr(lg_orch_module, attr_name, None)
                if lg_entry_fn is not None:
                    break
            if lg_entry_fn is None:
                for name in dir(lg_orch_module):
                    obj = getattr(lg_orch_module, name)
                    if callable(obj) and not name.startswith("_"):
                        lg_entry_fn = obj
                        break
            lg_sub_worker_fns: dict[str, object] = {}
            lg_sub_workers_dir = lg_output_dir / "sub_workers"
            if lg_sub_workers_dir.exists():
                for py_file in sorted(lg_sub_workers_dir.glob("*.py")):
                    mod = _load_generated_module(py_file)
                    fn_name = py_file.stem
                    fn = getattr(mod, fn_name, None)
                    if fn is not None:
                        lg_sub_worker_fns[fn_name] = fn
            lg_param_infos, _, _ = l3_generate._get_metadata()

            l3_generate_chip_callables = lg_chip_callables
            l3_generate_entry_fn = lg_entry_fn
            l3_generate_sub_worker_fns = lg_sub_worker_fns
            l3_generate_dc = lg_dc
            l3_generate_platform = l3_generate.platform
            l3_generate_runtime_name = lg_runtime_name
            l3_generate_param_infos = lg_param_infos
            _mark("l3_extract_artifacts")

        lm_head_weight = model.lm_head
        if padded_vocab != lm_head_weight.shape[0]:
            pad_rows = padded_vocab - lm_head_weight.shape[0]
            padding = torch.zeros(
                (pad_rows, lm_head_weight.shape[1]),
                dtype=lm_head_weight.dtype,
                device=lm_head_weight.device,
            )
            lm_head_weight = torch.cat([lm_head_weight, padding], dim=0)
        padded_lm_head_weight = lm_head_weight.to(torch.bfloat16).contiguous().cpu()
        _mark("pad_lm_head")
        layers = []
        for layer in model.layers:
            layers.append(self._kernel_layer_weights(layer))
            self._release_layer_weights(layer)
        final_norm_weight = model.final_norm_weight.view(1, -1).float().cpu()
        _mark("kernel_layer_weights")

        decode_weights = self._stack_decode_weights(layers)
        _mark("stack_decode_weights")

        timer.report()

        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            final_rms=final_rms,
            lm_head=lm_head,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            layers=layers,
            decode_weights=decode_weights,
            stacked_weights=stacked_weights,
            l3_generate_chip_callables=l3_generate_chip_callables,
            l3_generate_entry_fn=l3_generate_entry_fn,
            l3_generate_sub_worker_fns=l3_generate_sub_worker_fns,
            l3_generate_dc=l3_generate_dc,
            l3_generate_platform=l3_generate_platform,
            l3_generate_runtime_name=l3_generate_runtime_name,
            l3_generate_param_infos=l3_generate_param_infos,
        )

    @staticmethod
    def _stack_decode_weights(layers: list[_KernelLayerWeights]) -> dict[str, torch.Tensor]:
        """Stack per-layer weights into fused decode-kernel tensors."""
        # Stack from already-prepared per-layer kernel weights. Each
        # _KernelLayerWeights field is already in the kernel-ready shape/dtype
        # (transposed bf16 cpu for projections, [1, N] float cpu for norms),
        # so a plain cat along dim 0 is all that's left. Reading from the
        # original model.layers here would crash because _release_layer_weights
        # has already replaced those tensors with torch.empty(0).
        def cat(attr: str) -> torch.Tensor:
            return torch.cat([getattr(l, attr) for l in layers], dim=0)

        return {
            "decode_input_rms_weight": cat("input_rms_weight").contiguous(),
            "decode_wq":               cat("wq"),
            "decode_wk":               cat("wk"),
            "decode_wv":               cat("wv"),
            "decode_q_norm_weight":    cat("q_norm_weight").contiguous(),
            "decode_k_norm_weight":    cat("k_norm_weight").contiguous(),
            "decode_wo":               cat("wo"),
            "decode_post_rms_weight":  cat("post_rms_weight").contiguous(),
            "decode_w_gate":           cat("w_gate"),
            "decode_w_up":             cat("w_up"),
            "decode_w_down":           cat("w_down"),
        }

    @classmethod
    def _validate_total_kv_pages(cls, model: RuntimeModel, kernel_batch: int) -> None:
        """Validate that runtime KV page count matches compiled capacity."""
        if model.runtime.total_kv_pages is None:
            return
        expected_pages = kernel_batch * (model.runtime.max_seq_len + model.runtime.page_size - 1) // model.runtime.page_size
        if model.runtime.total_kv_pages != expected_pages:
            raise ValueError(
                "PyPTO Qwen3-14B kernels require total_kv_pages to match the runtime batch capacity: "
                f"{model.runtime.total_kv_pages} provided, {expected_pages} required."
            )

    @staticmethod
    def _kernel_weight(weight: torch.Tensor) -> torch.Tensor:
        """Convert a 2-D model weight into kernel-ready orientation and dtype."""
        return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu()

    @classmethod
    def _kernel_layer_weights(cls, layer) -> _KernelLayerWeights:
        """Convert one Hugging Face layer into kernel-ready weight tensors."""
        return _KernelLayerWeights(
            input_rms_weight=layer.input_rms_weight.view(1, -1).float().cpu(),
            wq=cls._kernel_weight(layer.wq),
            wk=cls._kernel_weight(layer.wk),
            wv=cls._kernel_weight(layer.wv),
            q_norm_weight=layer.q_norm_weight.view(1, -1).float().cpu(),
            k_norm_weight=layer.k_norm_weight.view(1, -1).float().cpu(),
            wo=cls._kernel_weight(layer.wo),
            post_rms_weight=layer.post_rms_weight.view(1, -1).float().cpu(),
            w_gate=cls._kernel_weight(layer.w_gate),
            w_up=cls._kernel_weight(layer.w_up),
            w_down=cls._kernel_weight(layer.w_down),
        )

    @staticmethod
    def _release_layer_weights(layer) -> None:
        """Drop original layer tensors after kernel-ready copies are built."""
        empty = torch.empty(0)
        layer.input_rms_weight = empty
        layer.wq = empty
        layer.wk = empty
        layer.wv = empty
        layer.q_norm_weight = empty
        layer.k_norm_weight = empty
        layer.wo = empty
        layer.post_rms_weight = empty
        layer.w_gate = empty
        layer.w_up = empty
        layer.w_down = empty

    @staticmethod
    def _validate_supported_shape(model: RuntimeModel) -> None:
        """Ensure the loaded model matches the bundled Qwen3-14B kernels."""
        config = model.config
        expected = {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
        }
        if actual != expected:
            mismatch = ", ".join(f"{k}={actual[k]} (expected {v})" for k, v in expected.items() if actual[k] != v)
            raise ValueError(
                "Bundled kernels under model/ currently support Qwen3-14B layer shapes only: " + mismatch
            )
        if model.runtime.page_size != _QWEN14B_PAGE_SIZE:
            raise ValueError(
                "PyPTO Qwen3-14B kernels require runtime page_size "
                f"{_QWEN14B_PAGE_SIZE}, got {model.runtime.page_size}."
            )
