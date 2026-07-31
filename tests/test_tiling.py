import torch

from styleforge.tiling import _linear_blend_weight, tiled_inference


class TestLinearBlendWeight:
    def test_shape(self):
        w = _linear_blend_weight(64, 64, torch.device("cpu"))
        assert w.shape == (64, 64)

    def test_center_higher_than_edge(self):
        w = _linear_blend_weight(64, 64, torch.device("cpu"))
        center = w[32, 32].item()
        edge = w[0, 0].item()
        assert center > edge


class TestTiledInference:
    def test_small_image_passthrough(self):
        img = torch.randn(1, 3, 256, 256)
        result = tiled_inference(img, lambda x: x * 2)
        assert torch.allclose(result, img * 2)

    def test_large_image_output_shape(self):
        img = torch.randn(1, 3, 2048, 2048)
        result = tiled_inference(img, lambda x: x, tile_size=512, overlap=64)
        assert result.shape == img.shape

    def test_large_image_approx_identity(self):
        img = torch.randn(1, 3, 1500, 1500)
        result = tiled_inference(img, lambda x: x, tile_size=512, overlap=64)
        assert torch.allclose(result, img, atol=1e-4)
