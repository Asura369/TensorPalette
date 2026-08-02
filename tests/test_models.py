import os

import pytest
import torch

from styleforge.cin import CINTransformer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def require_model_path(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        if os.environ.get("STYLEFORGE_REQUIRE_MODEL") == "1":
            pytest.fail(f"Model file {path} missing (required by STYLEFORGE_REQUIRE_MODEL)")
        pytest.skip(f"CIN model file {name} not found")
    return path


def test_cin_model_load_and_infer():
    path = require_model_path("multistyle.pth")

    model = CINTransformer(num_styles=5)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    style_ids = torch.tensor([0])
    with torch.no_grad():
        out = model(x, style_ids)

    assert out.shape == (1, 3, 256, 256)
    assert torch.isfinite(out).all()


def test_cin_model_interpolation():
    path = require_model_path("multistyle.pth")

    model = CINTransformer(num_styles=5)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model.forward_interpolated(x, style_id_a=0, style_id_b=1, alpha=0.5)

    assert out.shape == (1, 3, 256, 256)
    assert torch.isfinite(out).all()
