import pytest
import torch

from styleforge.transformer import StyleTransformer


class TestStyleTransformer:
    @pytest.mark.parametrize("size", [256, 512])
    def test_output_shape(self, size):
        model = StyleTransformer()
        model.eval()
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, size, size)

    def test_batch_output(self):
        model = StyleTransformer()
        model.eval()
        x = torch.randn(4, 3, 256, 256)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, 3, 256, 256)
