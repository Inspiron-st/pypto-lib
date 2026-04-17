# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pytest conftest for golden tests — adds repo root to sys.path.

Also installs stub ``pypto`` / ``pypto.ir`` / ``pypto.runtime`` modules when
the real pypto is not importable, so the golden unit tests can run in a
CPU-only CI job without building the compiler. The stubs expose the
attributes the tests patch (``pypto.ir.compile`` and
``pypto.runtime.execute_compiled``); if real pypto is installed it is used
as-is.
"""

import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _install_pypto_stubs() -> None:
    if importlib.util.find_spec("pypto") is not None:
        return

    import enum

    pypto = types.ModuleType("pypto")
    ir = types.ModuleType("pypto.ir")
    runtime = types.ModuleType("pypto.runtime")
    backend = types.ModuleType("pypto.backend")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "stub pypto: this function must be patched in tests"
        )

    class BackendType(enum.Enum):
        Ascend910B = "Ascend910B"
        Ascend950 = "Ascend950"

    ir.compile = _unavailable
    runtime.execute_compiled = _unavailable
    backend.BackendType = BackendType
    pypto.ir = ir
    pypto.runtime = runtime
    pypto.backend = backend

    sys.modules["pypto"] = pypto
    sys.modules["pypto.ir"] = ir
    sys.modules["pypto.runtime"] = runtime
    sys.modules["pypto.backend"] = backend


_install_pypto_stubs()
