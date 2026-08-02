import os

import pytest
import torch

from styleforge.vgg import Vgg16

VGG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vgg16.pth")


@pytest.fixture(scope="module")
def vgg():
    if not os.path.exists(VGG_PATH):
        if os.environ.get("STYLEFORGE_REQUIRE_MODEL") == "1":
            pytest.fail(f"VGG16 weights {VGG_PATH} missing (required by STYLEFORGE_REQUIRE_MODEL)")
        pytest.skip(f"VGG16 weights not found at {VGG_PATH}")
    return Vgg16(requires_grad=False)


def test_vgg_returns_relu4_2(vgg):
    """Regression: training content loss reads relu4_2, which used to be absent."""
    out = vgg(torch.randn(1, 3, 64, 64))
    assert hasattr(out, "relu4_2")
    assert out.relu4_2.shape == (1, 512, 8, 8)


def test_vgg_still_returns_relu4_3(vgg):
    """AdaIN depends on relu4_3; keep the field available."""
    out = vgg(torch.randn(1, 3, 64, 64))
    assert hasattr(out, "relu4_3")
    assert out.relu4_3.shape == (1, 512, 8, 8)
