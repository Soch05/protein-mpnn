import torch
from source.features import Cbeta, pairwise_distances, get_neighbours


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



def test_self_loop(k = 5):
    A = torch.randn(2, 10, 3) #BLL
    mask = torch.ones(2,10)
    ref = torch.arange(10)
    _,  E_idx = get_neighbours(A, mask, k=5)
    assert (E_idx[:,:,0] == ref).all()
    


