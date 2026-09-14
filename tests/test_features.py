import torch
from source.features import Cbeta, pairwise_distances


def test_cbeta_shape():
    X = torch.randn(2, 10, 4, 3)
    assert Cbeta(X).shape == (2, 10, 5, 3)


def test_distances_known_triangle():
    A = torch.tensor([[[0., 0., 0.], [3., 4., 0.]]])
    D = pairwise_distances(A, A)
    assert abs(D[0, 0, 1].item() - 5.0) < 1e-3
    assert D[0, 0, 0].item() < 1e-2


def test_distances_no_nan_backward():
    A = torch.randn(1, 5, 3, requires_grad=True)
    pairwise_distances(A, A).sum().backward()
    assert not torch.isnan(A.grad).any()