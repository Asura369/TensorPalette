import torch

from styleforge.utils import denormalize_batch, gram_matrix, normalize_batch


class TestGramMatrix:
    def test_shape(self):
        y = torch.randn(2, 64, 16, 16)
        gram = gram_matrix(y)
        assert gram.shape == (2, 64, 64)

    def test_symmetry(self):
        y = torch.randn(1, 32, 8, 8)
        gram = gram_matrix(y)
        assert torch.allclose(gram, gram.transpose(1, 2), atol=1e-6)

    def test_known_values(self):
        y = torch.ones(1, 2, 2, 2)
        gram = gram_matrix(y)
        expected_val = (2 * 2) / (2 * 2 * 2)
        assert torch.allclose(gram, torch.full((1, 2, 2), expected_val), atol=1e-6)


class TestNormalizeRoundTrip:
    def test_round_trip(self):
        original = torch.rand(2, 3, 64, 64) * 255
        normalized = normalize_batch(original.clone())
        recovered = denormalize_batch(normalized)
        assert torch.allclose(original, recovered, atol=1e-3)

    def test_normalized_range(self):
        batch = torch.full((1, 3, 4, 4), 128.0)
        normed = normalize_batch(batch)
        assert normed.mean().abs() < 1.0
