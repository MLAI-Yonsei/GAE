"""Checks for the GAE Step 1 subspace-matrix option. Run: python tests/test_gae_options.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gae import gae_torch

torch.manual_seed(0)
DT = torch.float64


def _projector(U):
    return U @ U.T


def _top_r_projector(C, r):
    evals, evecs = torch.linalg.eigh(C)
    return _projector(evecs[:, torch.argsort(evals, descending=True)[:r]])


def _data(N=400, d=8, k=12):
    # large mean along e_0 and the largest spread along e_1: the two matrices pick different subspaces
    H = torch.randn(N, d, dtype=DT) * torch.tensor([0.1, 3.0] + [1.0] * (d - 2), dtype=DT)
    H[:, 0] += 10.0
    B = torch.randn(d, k, dtype=DT)
    return H, B


def test_default_is_centered_covariance():
    H, B = _data()
    _, info_default = gae_torch(B, H, 1)
    _, info_centered = gae_torch(B, H, 1, subspace_matrix="centered")
    Hc = H - H.mean(0)
    P_ref = _top_r_projector(Hc.T @ Hc / (H.shape[0] - 1), 1)
    assert torch.allclose(_projector(info_default["U_ood"]), P_ref, atol=1e-10)
    assert torch.allclose(_projector(info_centered["U_ood"]), P_ref, atol=1e-10)


def test_second_moment_matches_paper_definition():
    H, B = _data()
    _, info = gae_torch(B, H, 1, subspace_matrix="second_moment")
    P_ref = _top_r_projector(H.T @ H / H.shape[0], 1)
    assert torch.allclose(_projector(info["U_ood"]), P_ref, atol=1e-10)
    # with a large mean the two options must disagree (mean direction vs spread direction)
    _, info_c = gae_torch(B, H, 1)
    assert (_projector(info["U_ood"]) - _projector(info_c["U_ood"])).norm() > 1.0


def test_step1_columns_stay_in_ood_subspace():
    H, B = _data()
    for mode in ("centered", "second_moment"):
        B_gae, info = gae_torch(B, H, 3, subspace_matrix=mode)
        P = _projector(info["U_ood"])
        assert torch.allclose(P @ B_gae, B_gae, atol=1e-10)


def test_invalid_option_raises():
    H, B = _data()
    try:
        gae_torch(B, H, 1, subspace_matrix="covariance")
    except AssertionError:
        return
    raise AssertionError("invalid subspace_matrix was accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
