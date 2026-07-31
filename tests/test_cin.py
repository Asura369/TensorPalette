import torch

from styleforge.cin import CINTransformer, ConditionalInstanceNorm2d


class TestConditionalInstanceNorm2d:
    def test_output_shape(self):
        cin = ConditionalInstanceNorm2d(num_styles=3, channels=64)
        x = torch.randn(2, 64, 16, 16)
        out = cin(x, style_id=0)
        assert out.shape == (2, 64, 16, 16)

    def test_different_styles_differ(self):
        cin = ConditionalInstanceNorm2d(num_styles=3, channels=64)
        with torch.no_grad():
            cin.gamma[1].fill_(2.0)
            cin.beta[1].fill_(0.5)
        x = torch.randn(1, 64, 8, 8)
        out0 = cin(x, style_id=0)
        out1 = cin(x, style_id=1)
        assert not torch.allclose(out0, out1)


class TestCINTransformer:
    def test_output_shape(self):
        model = CINTransformer(num_styles=5)
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y = model(x, style_id=0)
        assert y.shape == (1, 3, 256, 256)

    def test_different_styles_produce_different_output(self):
        model = CINTransformer(num_styles=3)
        model.eval()
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, ConditionalInstanceNorm2d):
                    module.gamma[1].fill_(2.0)
                    module.beta[1].fill_(0.5)
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            y0 = model(x, style_id=0)
            y1 = model(x, style_id=1)
        assert not torch.allclose(y0, y1, atol=1e-3)

    def test_get_set_style_params(self):
        model = CINTransformer(num_styles=2)
        params = model.get_style_params(style_id=0)
        assert len(params) > 0
        for name, p in params.items():
            assert "gamma" in p
            assert "beta" in p
