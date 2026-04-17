# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for golden.tensor_spec."""

import pytest
import torch
from golden.tensor_spec import TensorSpec


class TestTensorSpecCreateTensor:
    """Tests for TensorSpec.create_tensor() with various init_value strategies."""

    def test_none_init_creates_zeros(self):
        """init_value=None produces a zero-filled tensor."""
        spec = TensorSpec("x", [4, 8], torch.float32)
        t = spec.create_tensor()
        assert t.shape == (4, 8)
        assert t.dtype == torch.float32
        assert torch.equal(t, torch.zeros(4, 8, dtype=torch.float32))

    def test_int_init_creates_full(self):
        """init_value=int/float fills every element with that constant."""
        spec = TensorSpec("x", [3, 5], torch.float32, init_value=7)
        t = spec.create_tensor()
        assert torch.equal(t, torch.full((3, 5), 7, dtype=torch.float32))

    def test_tensor_init_uses_directly(self):
        """init_value=torch.Tensor uses the tensor directly, casting dtype."""
        data = torch.arange(0, 6, dtype=torch.float64).reshape(2, 3)
        spec = TensorSpec("x", [2, 3], torch.float32, init_value=data)
        t = spec.create_tensor()
        assert t.dtype == torch.float32
        assert torch.allclose(t, data.float())

    @pytest.mark.parametrize("factory", [torch.randn, torch.rand, torch.zeros, torch.ones])
    def test_torch_factory_callables(self, factory):
        """Each torch factory callable produces a tensor with correct shape/dtype."""
        spec = TensorSpec("x", [4, 4], torch.float32, init_value=factory)
        t = spec.create_tensor()
        assert t.shape == (4, 4)
        assert t.dtype == torch.float32

    def test_custom_callable(self):
        """init_value=custom_fn calls with no args and casts dtype."""
        def make_data():
            return torch.arange(0, 4, dtype=torch.float64)

        spec = TensorSpec("x", [4], torch.float32, init_value=make_data)
        t = spec.create_tensor()
        assert t.dtype == torch.float32
        assert torch.allclose(t, torch.arange(0, 4, dtype=torch.float32))

    def test_unsupported_init_value_raises(self):
        """Unsupported init_value type raises TypeError."""
        spec = TensorSpec("x", [4], torch.float32, init_value="invalid")
        with pytest.raises(TypeError, match="Unsupported init_value type"):
            spec.create_tensor()

    def test_is_output_flag(self):
        """is_output flag is stored correctly and defaults to False."""
        spec_in = TensorSpec("a", [4], torch.float32)
        spec_out = TensorSpec("b", [4], torch.float32, is_output=True)
        assert spec_in.is_output is False
        assert spec_out.is_output is True

    def test_tensor_init_ignores_spec_shape(self):
        """Pin current behavior: when init_value is a Tensor, spec.shape is NOT enforced.

        The spec says [2, 2] but the supplied tensor is [3]. ``create_tensor``
        returns the supplied tensor cast to dtype — shape mismatch is not
        validated. Kept as a regression pin; revisit if we decide to enforce
        shape matching in ``TensorSpec.create_tensor``.
        """
        data = torch.tensor([1.0, 2.0, 3.0])
        spec = TensorSpec("x", [2, 2], torch.float32, init_value=data)
        t = spec.create_tensor()
        assert t.shape == (3,)
        assert t.shape != tuple(spec.shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
