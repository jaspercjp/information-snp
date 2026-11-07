"""`InfoTheoryMetrics.info_RPC`, parallel over latitude rows.

    from info_rpc_parallel import info_RPC
    I_gf, I_sf, H_f, H_g = info_RPC(ensemble, signal, observation, n_jobs=16)

Each cell runs the thesis arithmetic unchanged (`freedman_diaconis_bins`, `I_jugaad`), so the
four arrays are bitwise equal to the serial `InfoTheoryMetrics.info_RPC` (histogram method;
tests/test_jugaad.py). Like the original, NaN members are zero-filled in `ensemble` in place.
"""
import os
from multiprocessing import get_context

import numpy as np

from InfoTheoryMetrics import I_jugaad, freedman_diaconis_bins

_G = {}


def _row(lat):
    f, s, g, alts = _G["f"], _G["s"], _G["g"], _G["alts"]
    N, ny = f.shape[0], f.shape[-1]
    out = np.zeros((4, len(alts), ny))
    for lon in range(ny):
        g_t, s_t, f_t = g[:, lat, lon], s[:, lat, lon], np.array(f[:, :, lat, lon])
        f_t[np.where(np.isnan(f_t))] = 0.0
        f_bins = freedman_diaconis_bins(f_t.flatten())
        freedman_diaconis_bins(s_t), freedman_diaconis_bins(g_t)       # as the original: may raise
        s_bins_ext = freedman_diaconis_bins(np.tile(s_t, N))
        g_bins_ext = freedman_diaconis_bins(np.tile(g_t, N))
        for k, a in enumerate(alts):
            bf = int(np.ceil(f_bins * a))
            I_gf, H_g, _ = I_jugaad(g_t, f_t, nbins_alt=(int(np.ceil(g_bins_ext * a)), bf))
            I_sf, _, H_f = I_jugaad(s_t, f_t, nbins_alt=(int(np.ceil(s_bins_ext * a)), bf))
            out[:, k, lon] = I_gf, I_sf, H_f, H_g
    return out


def info_RPC(ensemble, signal, observation, quant_uncert=True, n_jobs=None):
    """(I_gf, I_sf, H_f, H_g), each (n_alt, lat, lon); alts 0.5/1/1.5 if quant_uncert."""
    f = ensemble.psl.to_numpy()
    _G.update(f=f, s=signal.psl.to_numpy(), g=observation.psl.to_numpy(),
              alts=[0.5, 1, 1.5] if quant_uncert else [1])
    n_jobs = n_jobs or int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    if n_jobs > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            rows = pool.map(_row, range(f.shape[2]))
    else:
        rows = [_row(j) for j in range(f.shape[2])]
    _G.clear()
    f[np.isnan(f)] = 0.0                                             # the original's side effect
    return tuple(np.stack([r[i] for r in rows], axis=1) for i in range(4))
