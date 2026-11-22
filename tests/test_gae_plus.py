"""Self-check for the GAE+ additions. Run: python tests/test_gae_plus.py"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gae import gae_torch, ledoit_wolf_ridge, shrink_covariance
from gae_plus import ETA_GRID, gate, mix, select, utility

torch.manual_seed(0)
d, k = 16, 64
B = torch.randn(d, k)
H_big = torch.randn(4000, d) @ torch.randn(d, d)
H_small = H_big[:24]
center = lambda H: H - H.mean(0)

# Step 1 shrinkage: off reproduces main; the intensity lies in [0, 1] and grows when samples are few
W_plain, _ = gae_torch(B, H_small, 4)
W_off, info = gae_torch(B, H_small, 4, shrink=None)
assert torch.equal(W_plain, W_off) and info["shrink_rho"] is None
rho_small = shrink_covariance(center(H_small), B=B)[1]
rho_big = shrink_covariance(center(H_big), B=B)[1]
assert 0 <= rho_big < rho_small <= 1
assert 0 < shrink_covariance(center(H_small), H_id=H_big)[1] <= 1
for mode, kw in (("decoder", {}), ("id", dict(H_id=H_big))):
    W, info = gae_torch(B, H_small, 4, shrink=mode, **kw)
    assert W.shape == B.shape and 0 <= info["shrink_rho"] <= 1
assert ledoit_wolf_ridge(center(torch.randn(32, k).relu())) > 0

# eta grid: the corners are A0, A1, A2 and every point is a convex mix
A = [(torch.full((2, 2), float(i)), torch.full((2,), float(i))) for i in range(3)]
assert torch.equal(mix(*A, (0, 0))[0], A[0][0]) and torch.equal(mix(*A, (1, 0))[0], A[1][0]) and torch.equal(mix(*A, (0, 1))[1], A[2][1])
assert all(e1 >= 0 and e2 >= 0 and e1 + e2 <= 1 for e1, e2 in ETA_GRID)

# utility: Fixed scores 0, a better decoder scores > 0, a decoder whose margins collapse falls back to Fixed
row = lambda s0, s_empty, drop: dict(s0=s0, s_empty=s_empty, s_del=[s0 - drop] * 8)
fixed = [row(5, 4, 0.5), row(5, 4, 0.5), row(5, 4.95, 0.01)]  # third input: Fixed margin <= tau, ignored
better = [row(5, 4, 0.8), row(5, 4, 0.8), row(5, 4, 0.8)]
broken = [row(5, 4.99, 3.0), row(5, 4.99, 3.0), row(5, 4, 0.8)]
assert abs(utility(fixed, fixed)) < 1e-12 and abs(utility(better, fixed) - 0.6) < 1e-9 and abs(utility(broken, fixed)) < 1e-12
assert gate(broken, fixed)[0] is fixed[0] and gate(better, fixed)[0] is better[0]
cands = [dict(eta=(0.0, 0.0), rows=fixed), dict(eta=(0.0, 1.0), rows=better), dict(eta=(1.0, 0.0), rows=broken)]
assert select(cands, fixed)["eta"] == (0.0, 1.0)
assert select([cands[0], cands[2]], fixed)["eta"] == (0.0, 0.0)  # tie -> the smaller change
print("ok")
