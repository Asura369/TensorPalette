import numpy as np
import torch
from PIL import Image

from styleforge.cin import CINTransformer
from styleforge.engine import InferenceEngine
from styleforge.tiling import tiled_inference


class TestInferenceEngine:
    def _make_cin_model(self, num_styles=3):
        model = CINTransformer(num_styles=num_styles)
        model.eval()
        with torch.no_grad():
            for _, module in model.named_modules():
                from styleforge.cin import ConditionalInstanceNorm2d
                if isinstance(module, ConditionalInstanceNorm2d):
                    module.gamma[1].fill_(2.0)
                    module.beta[1].fill_(0.5)
                    module.gamma[2].fill_(0.5)
                    module.beta[2].fill_(-0.3)
        return model

    def test_stylize_single_style(self, tmp_path):
        model = self._make_cin_model()
        model_path = str(tmp_path / "cin.pth")
        torch.save(model.state_dict(), model_path)

        engine = InferenceEngine(device=torch.device("cpu"))
        engine.load_cin("test_cin", model_path, num_styles=3)

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = engine.stylize(img, "test_cin", style_id=0)
        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)

    def test_style_interpolation(self, tmp_path):
        model = self._make_cin_model()
        model_path = str(tmp_path / "cin.pth")
        torch.save(model.state_dict(), model_path)

        engine = InferenceEngine(device=torch.device("cpu"))
        engine.load_cin("test_cin", model_path, num_styles=3)

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result_a = engine.stylize(img, "test_cin", style_a=0, style_b=1, alpha=0.0)
        result_b = engine.stylize(img, "test_cin", style_a=0, style_b=1, alpha=1.0)
        result_mid = engine.stylize(img, "test_cin", style_a=0, style_b=1, alpha=0.5)

        arr_a = np.array(result_a)
        arr_b = np.array(result_b)
        arr_mid = np.array(result_mid)

        assert not np.allclose(arr_a, arr_b)
        assert not np.allclose(arr_a, arr_mid)

    def test_stylize_large_image_uses_tiling(self, tmp_path):
        model = self._make_cin_model()
        model_path = str(tmp_path / "cin.pth")
        torch.save(model.state_dict(), model_path)

        engine = InferenceEngine(device=torch.device("cpu"))
        engine.load_cin("test_cin", model_path, num_styles=3)

        arr = np.random.randint(0, 255, (1300, 1300, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = engine.stylize(img, "test_cin", style_id=1)
        assert result.size == (1300, 1300)

        content = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        style_ids = torch.tensor([1])
        expected = tiled_inference(content, lambda tile: model(tile, style_ids)).clamp(0, 255)
        got = torch.tensor(np.array(result), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        assert torch.allclose(got, expected, atol=1.0)

    def test_stylize_large_image_interp_uses_tiling(self, tmp_path):
        model = self._make_cin_model()
        model_path = str(tmp_path / "cin.pth")
        torch.save(model.state_dict(), model_path)

        engine = InferenceEngine(device=torch.device("cpu"))
        engine.load_cin("test_cin", model_path, num_styles=3)

        arr = np.random.randint(0, 255, (1300, 1300, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        result = engine.stylize(img, "test_cin", style_a=0, style_b=2, alpha=0.5)
        assert result.size == (1300, 1300)

        content = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        expected = tiled_inference(
            content, lambda tile: model.forward_interpolated(tile, 0, 2, 0.5)
        ).clamp(0, 255)
        got = torch.tensor(np.array(result), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        assert torch.allclose(got, expected, atol=1.0)
