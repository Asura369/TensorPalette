import torch

from styleforge.train_cin import split_dataset


class FakeDataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.zeros(3, 4, 4), 0


def test_split_dataset_returns_disjoint_subsets():
    train, val = split_dataset(FakeDataset(100), val_fraction=0.02, seed=42)
    assert len(val) == 2
    assert len(train) == 98
    assert set(train.indices).isdisjoint(set(val.indices))


def test_split_dataset_is_deterministic():
    t1, v1 = split_dataset(FakeDataset(1000), val_fraction=0.02, seed=7)
    t2, v2 = split_dataset(FakeDataset(1000), val_fraction=0.02, seed=7)
    assert len(v1) == len(v2) == 20
    assert t1.indices == t2.indices
    assert v1.indices == v2.indices


def test_split_dataset_zero_fraction_means_no_val():
    train, val = split_dataset(FakeDataset(100), val_fraction=0.0, seed=42)
    assert val is None
    assert len(train) == 100
