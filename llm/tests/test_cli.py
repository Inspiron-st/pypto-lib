# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm.cli.main as cli
from llm.core import GenerateConfig, RuntimeConfig


class _FakeEngine:
    instances: list["_FakeEngine"] = []

    def __init__(self, kv_cache_manager=None, executor=None):
        self.kv_cache_manager = kv_cache_manager
        self.executor = executor
        self.init_calls = []
        self.result_prompts = []
        self.stream_prompts = []
        _FakeEngine.instances.append(self)

    def init_model(self, **kwargs):
        self.init_calls.append(kwargs)

    def generate_result(self, model_id, prompt, config):
        self.result_prompts.append((model_id, prompt, config))
        return SimpleNamespace(
            text=f"generated:{prompt}",
            token_ids=[1, 2, 3],
            finish_reason="length",
        )

    def generate(self, model_id, prompt, config):
        self.stream_prompts.append((model_id, prompt, config))
        return iter(["stream:", prompt])


class _FakeNpuExecutor:
    instances: list["_FakeNpuExecutor"] = []

    def __init__(self, kv_cache_manager, **kwargs):
        self.kv_cache_manager = kv_cache_manager
        self.kwargs = kwargs
        _FakeNpuExecutor.instances.append(self)


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "serving.json"
    path.write_text(json.dumps(data))
    return path


def _base_config(model_dir: Path, *, backend: str = "cpu") -> dict:
    return {
        "model": {
            "model_id": "test-model",
            "model_dir": str(model_dir),
            "model_format": "huggingface",
        },
        "runtime": {
            "backend": backend,
            "max_seq_len": 128,
        },
        "generation": {
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": ["</s>"],
        },
    }


def test_load_serving_config_defaults_npu_page_size(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = _write_config(tmp_path, _base_config(model_dir, backend="npu"))

    config = cli.load_serving_config(config_path)

    assert config.backend == "npu"
    assert config.runtime.page_size == 256
    assert config.runtime.max_seq_len == 128
    assert config.generation.stop == ("</s>",)


def test_load_serving_config_rejects_missing_model_dir(tmp_path):
    config_path = _write_config(
        tmp_path,
        {
            "model": {},
            "runtime": {"backend": "cpu"},
            "generation": {},
        },
    )

    with pytest.raises(ValueError, match="model.model_dir"):
        cli.load_serving_config(config_path)


def test_main_one_shot_cpu_uses_json_config(tmp_path, monkeypatch, capsys):
    _FakeEngine.instances.clear()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = _write_config(tmp_path, _base_config(model_dir, backend="cpu"))
    monkeypatch.setattr(cli, "LLMEngine", _FakeEngine)

    assert cli.main(["--config", str(config_path), "--prompt", "hello"]) == 0

    engine = _FakeEngine.instances[-1]
    assert engine.executor is None
    assert engine.init_calls[0]["model_id"] == "test-model"
    assert engine.init_calls[0]["model_dir"] == str(model_dir.resolve())
    assert engine.result_prompts[0][1] == "hello"
    out = capsys.readouterr().out
    assert "text: generated:hello" in out
    assert "finish_reason: length" in out


def test_main_suppresses_startup_logs_by_default(tmp_path, monkeypatch):
    _FakeEngine.instances.clear()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = _write_config(tmp_path, _base_config(model_dir, backend="cpu"))
    startup_context_flags = []
    monkeypatch.setattr(cli, "LLMEngine", _FakeEngine)
    monkeypatch.setattr(
        cli,
        "_startup_log_context",
        lambda *, enabled: _RecordingContext(startup_context_flags, enabled),
    )

    cli.main(["--config", str(config_path), "--prompt", "hello"])

    assert startup_context_flags == [True]


def test_main_can_show_startup_logs(tmp_path, monkeypatch):
    _FakeEngine.instances.clear()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = _write_config(tmp_path, _base_config(model_dir, backend="cpu"))
    startup_context_flags = []
    monkeypatch.setattr(cli, "LLMEngine", _FakeEngine)
    monkeypatch.setattr(
        cli,
        "_startup_log_context",
        lambda *, enabled: _RecordingContext(startup_context_flags, enabled),
    )

    cli.main(["--config", str(config_path), "--prompt", "hello", "--show-startup-logs"])

    assert startup_context_flags == [False]


def test_create_engine_npu_wires_executor_options(tmp_path, monkeypatch):
    _FakeEngine.instances.clear()
    _FakeNpuExecutor.instances.clear()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_data = _base_config(model_dir, backend="npu")
    config_data["npu"] = {
        "platform": "a5",
        "device_id": 3,
        "save_kernels_dir": "/tmp/kernels",
        "pypto_root": "/tmp/pypto",
    }
    config_path = _write_config(tmp_path, config_data)
    monkeypatch.setattr(cli, "LLMEngine", _FakeEngine)
    monkeypatch.setattr(cli, "PyptoQwen14BExecutor", _FakeNpuExecutor)

    engine = cli.create_engine(cli.load_serving_config(config_path))

    executor = _FakeNpuExecutor.instances[-1]
    assert engine.executor is executor
    assert executor.kwargs == {
        "pypto_root": "/tmp/pypto",
        "platform": "a5",
        "device_id": 3,
        "save_kernels_dir": "/tmp/kernels",
    }


def test_run_interactive_reuses_engine_until_exit(capsys):
    _FakeEngine.instances.clear()
    engine = _FakeEngine()
    config = cli.ServingConfig(
        model=cli.ModelCliConfig(
            model_id="test-model",
            model_dir="/tmp/model",
            model_format="huggingface",
            loader_options={},
        ),
        runtime=RuntimeConfig(),
        generation=GenerateConfig(max_new_tokens=1),
        backend="cpu",
        npu=cli.NpuCliConfig(),
    )
    prompts = iter(["/help", "/config", "/clear", "first", "", "second", "/exit"])

    cli.run_interactive(engine, config, input_fn=lambda _: next(prompts))

    assert [prompt for _, prompt, _ in engine.result_prompts] == ["first", "second"]
    out = capsys.readouterr().out
    assert "PyPTO Serving interactive generation" in out
    assert "Commands: /help, /config, /clear, /exit, /quit" in out
    assert "Commands:" in out
    assert "model_id=test-model" in out
    assert "--- new prompt session ---" in out
    assert "[assistant]" in out
    assert "text: generated:first" in out
    assert "text: generated:second" in out
    assert "Bye." in out


class _RecordingContext:
    def __init__(self, flags: list[bool], enabled: bool) -> None:
        self._flags = flags
        self._enabled = enabled

    def __enter__(self):
        self._flags.append(self._enabled)

    def __exit__(self, exc_type, exc, traceback):
        return False
