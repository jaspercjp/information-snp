"""Non-parametric significance for decadal skill and the signal-to-noise paradox.

    import sys; sys.path.insert(0, "../scripts")
    import significance_test_RPC as T, dcpp_handles as G, depresys4_handles as D

    r = T.test(G.get(lead="13-60", var="TREFHT"))     # any handle, pooled or single
    r = T.test(D.get(lead="13-60", var="SLP"), rho_m="eade")
    r.report()

This is Eade et al. (2014)'s procedure, replicated
--------------------------------------------------
Their five steps, and where each one lives:

  1  resample T=J validation cases WITH replacement, in blocks of L=5 consecutive
     years, to take autocorrelation into account            `_boot_idx`, `_weights`
  2  resample M-3 ensemble members WITHOUT replacement, independently per replicate;
     no replacement over members because "repeatedly resampling the same members
     reduces the number of independent data points in the ensemble mean, and so
     reduces the correlation unfairly"                      `_eade_boot`, step 2
  3  compute the statistic from THAT sample's ensemble mean  `_eade_boot`, step 3
  4  repeat 1000 times to build a pdf                        `B=1000`
  5  two-tailed test: reject "RPC is not different to one" at the 90% level if the
     5-95% confidence interval does not span one            `sig_rpc`, `RPC_gm_ci`

`sig_skill`/`sig_rpc` are exactly step 5 on rho_o vs 0 and RPC vs 1. `sig_rpc_gt1` and
`sig_rpc_lt1` split the two-tailed rejection by direction, since only the upper tail is
the signal-to-noise paradox. `RPC_gm_ci` applies step 5 to the area-weighted global-mean
RPC -- one test per configuration, so no multiple-testing question arises there at all.

What step 2 does and does not do
--------------------------------
It puts finite-M *uncertainty* inside the interval. It does not remove the finite-M
*bias*: an M-3 member ensemble mean is noisier than the M-member one, so the pdf is
centred BELOW the point estimate, and increasingly so as M shrinks (M-3 is 70% of a
10-member ensemble but 94% of a 46-member one). `med_offset` and `boot_med` expose that
offset per configuration, and `ci_widen` reports how much wider step 2 makes the interval
than resampling start dates alone -- so the cost and the benefit are both visible.

Two additions, off by default, never substituted for the above (`extras=True`)
------------------------------------------------------------------------------
`RPC_pm` -- the perfect-model RPC at this (N, J). For a perfect model with predictable
fraction p, rho_o = p^2/sqrt(p^2+(1-p^2)/N), so RPC = p/sqrt(p^2+(1-p^2)/N) < 1: at
N=46, p=0.33 a perfect model scores 0.921 (debiased) or 0.849 (Eade's rho_m), not 1.
Step 2 does not address this, because it is a bias in the statistic, not a width. Holding
each member out as pseudo-observations measures it, and also gives the null floor on
`frac(RPC>1)`, which is large (~0.3-0.6) purely from cell-to-cell scatter at J~50.
`p_perm`/`p_field_skill` -- a block-permutation null and Livezey-Chen field significance
for skill, which test independence directly rather than by CI inversion.

Blocks
------
Consecutive lead 13-60 windows are 48-month means offset by 12 months, so neighbours
share 3 of their 4 years, exactly the case Eade's blocks of five address. Dropping to
L=1 is badly liberal -- reproduced in `test_significance_test_RPC.py` §3.

Multiple testing
----------------
Eade apply no correction, so `area_skill` / `area_rpc` -- the step-5 areas -- are the
directly comparable numbers and are the headline. `area_*_fdr` adds Benjamini-Hochberg at
`q` over the same pdf, because on thousands of correlated cells an uncorrected pointwise
map overstates how much is really there. Both are reported; neither replaces the other.

Eade's 11.25x12.5 deg smoothing is ON by default in every handle, so adjacent cells are
strongly correlated and the pointwise count is not a count of independent tests.
Negative-correlation cells are masked before any area fraction ("they imply zero
skill"); `mask_min=0.2` reproduces this project's older numbers instead.
"""

import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

import sys
import warnings
import numpy as np


def _nan(fn, *a, **kw):
    """A nan-aware reducer without the all-NaN-slice warning; masked cells ARE all-NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **kw)

# --------------------------------------------------------------------- estimators
# Vectorized twins of smyle_metrics.pearson_coeff / ensemble_SNR{,_debiased}, checked
# equal to floating point in test_significance_test_RPC.py §1. They live here so the
# bootstrap can run them thousands of times off precomputed sums.


def _corr(s, G):
    """Pearson correlation along axis 0, per cell."""
    num = np.mean((s - s.mean(0)) * (G - G.mean(0)), axis=0)
    return num / (s.std(0) * G.std(0))


def _rho_m(F, kind="debiased"):
    """(N,J,...) -> rho_m. "eade" is sd(s)/<sd(member)>; "debiased" removes the 1/N."""
    N = F.shape[0]
    v = F.var(axis=1)
    vs = F.mean(0).var(0)
    if kind == "eade":
        return np.sqrt(vs) / np.sqrt(v).mean(0)
    return np.sqrt(np.maximum((N * vs - v.mean(0)) / (N - 1), 0.0) / v.mean(0))


def _boot_stats(F, G, Wt, kind="debiased"):
    """(rho_o, rho_m) for each row of `Wt`, a (B, J) matrix of resample weights.

    A resample of the start-date axis, with or without replacement, enters every
    estimator only through how many times each start date is drawn. So a whole
    bootstrap replicate is two BLAS matmuls against precomputed first and second
    moments -- no (N, J, lat, lon) copy per replicate.
    """
    N, J = F.shape[:2]
    sh = F.shape[2:]
    F2 = F.reshape(N, J, -1)
    F2 = F2 - F2.mean(1, keepdims=True)          # exact in these formulae, and stops
    G2 = G.reshape(J, -1); G2 = G2 - G2.mean(0)  # m2 - m1^2 cancelling catastrophically
    s2 = F2.mean(0)
    m1 = np.tensordot(Wt, F2, axes=(1, 1))       # (B, N, cells)
    v = np.tensordot(Wt, F2 ** 2, axes=(1, 1)) - m1 ** 2
    sm1 = Wt @ s2; svs = (Wt @ s2 ** 2) - sm1 ** 2
    gm1 = Wt @ G2; gvs = (Wt @ G2 ** 2) - gm1 ** 2
    rho_o = ((Wt @ (s2 * G2)) - sm1 * gm1) / np.sqrt(svs * gvs)
    if kind == "eade":
        rho_m = np.sqrt(np.maximum(svs, 0)) / np.sqrt(np.maximum(v, 0)).mean(1)
    else:
        tot2 = v.mean(1)
        rho_m = np.sqrt(np.maximum((N * svs - tot2) / (N - 1), 0.0) / tot2)
    return rho_o.reshape((-1,) + sh), rho_m.reshape((-1,) + sh)


def _eade_boot(F, G, L, B, rng, kind, drop=3, chunk=32, Wsp=None, good=None):
    """Eade et al. (2014) steps 1-4, verbatim. -> (rho_o, rho_m, RPC, RPC_gm), (B, cells).

    1  resample T=J validation cases WITH replacement, in blocks of L=5 consecutive years
    2  resample M-`drop` ensemble members WITHOUT replacement, independently per replicate
       ("replacement is not used over ensemble members because repeatedly resampling the
       same members reduces the number of independent data points in the ensemble mean,
       and so reduces the correlation unfairly")
    3  compute the statistic from THAT sample's ensemble mean
    4  B=1000 replicates -> a pdf

    Step 2 is what makes this different from resampling start dates alone: the ensemble
    mean is rebuilt from M-drop members every replicate, so finite-M uncertainty is
    inside the spread. It does NOT remove the finite-M *bias* -- an M-drop ensemble mean
    is noisier than the M-member one, so the pdf is centred BELOW the point estimate,
    increasingly so as M shrinks. `boot_med` in the result exposes that offset.

    `Wsp`/`good` (flattened area weights and validity mask) additionally return the pdf
    of the global-mean RPC -- one test per configuration, so no multiple-testing problem.
    """
    N, J = F.shape[:2]
    M = N - drop
    if M < 2:
        raise ValueError(f"N={N} leaves M-{drop}={M} members; lower `drop`")
    Wt = _weights(block_idx_years(np.arange(J), L, rng, B), J)      # step 1
    Pm = np.zeros((B, N))
    for b in range(B):                                              # step 2
        Pm[b, rng.choice(N, M, replace=False)] = 1.0 / M
    return _eade_boot_at(F, G, Wt, Pm, kind, chunk, Wsp, good)


def _eade_boot_at(F, G, Wt, Pm, kind="debiased", chunk=32, Wsp=None, good=None):
    """Steps 3-4 for given step-1 weights `Wt` (B,J) and step-2 selections `Pm` (B,N).

    Split out from `_eade_boot` so a replicate can be checked against an explicitly
    built subsample-then-resample, which is what test §1b does.
    """
    N, J = F.shape[:2]
    M = int(round(1.0 / Pm[Pm > 0][0]))
    B = Wt.shape[0]
    F2 = F.reshape(N, J, -1)
    F2 = F2 - F2.mean(1, keepdims=True)
    Fsq = F2 ** 2
    G2 = G.reshape(J, -1); G2 = G2 - G2.mean(0)
    C = F2.shape[-1]
    o = np.empty((B, C)); m = np.empty((B, C))
    for a in range(0, B, chunk):
        w, P = Wt[a:a + chunk], Pm[a:a + chunk]
        m1 = np.tensordot(w, F2, axes=(1, 1))                       # (b, N, cells)
        v = np.tensordot(w, Fsq, axes=(1, 1)) - m1 ** 2
        sb = np.tensordot(P, F2, axes=(1, 0))                       # (b, J, cells) step 3
        sm1 = (w[:, :, None] * sb).sum(1)
        vs = (w[:, :, None] * sb ** 2).sum(1) - sm1 ** 2
        gm1 = w @ G2; gvs = (w @ G2 ** 2) - gm1 ** 2
        cov = (w[:, :, None] * (sb * G2[None])).sum(1) - sm1 * gm1
        o[a:a + chunk] = cov / np.sqrt(vs * gvs)
        if kind == "eade":                     # mean over the SELECTED members only
            m[a:a + chunk] = (np.sqrt(np.maximum(vs, 0))
                              / np.einsum("bn,bnc->bc", P, np.sqrt(np.maximum(v, 0))))
        else:
            tot2 = np.einsum("bn,bnc->bc", P, v)
            m[a:a + chunk] = np.sqrt(np.maximum((M * vs - tot2) / (M - 1), 0.0) / tot2)
    rgm = None
    if Wsp is not None:
        k = good & np.isfinite(o).all(0) & np.isfinite(m).all(0)
        wk = Wsp[k] / Wsp[k].sum()
        rgm = (o[:, k] @ wk) / (m[:, k] @ wk)
    with np.errstate(invalid="ignore", divide="ignore"):
        return o, m, o / m, rgm


def year_windows(Y, years=10, full_only=True):
    """Index sets for every window of `years` CONSECUTIVE CALENDAR YEARS present in Y.

    Indexing by array POSITION is wrong here: the pooled start years have gaps (the
    models' start sets are intersected), so 5 consecutive positions span up to 11
    calendar years. Windows are also not circular -- wrapping would join 2014 to 1960.
    """
    Y = np.asarray(Y)
    out = []
    for y0 in range(int(Y.min()), int(Y.max()) - years + 2):
        k = np.where((Y >= y0) & (Y < y0 + years))[0]
        if len(k) == years or (not full_only and len(k) >= 4):
            out.append(k)
    return out


def block_idx_years(Y, L, rng, B):
    """Eade step 1: T=J cases WITH replacement, in blocks of L consecutive CALENDAR years.

    Blocks are the fully-populated L-year windows of `Y`, drawn with replacement,
    concatenated and truncated to exactly T=J cases. Non-circular, so no block joins the
    end of the record to its start.

    Indexing by array POSITION instead is wrong and biases toward significance. The
    pooled start years have gaps (`dcpp_handles` intersects the models' start sets --
    pooled SLP is missing 1976 and 2005-2010), so 8 of the 48 position-blocks bundle
    start dates 7-11 years apart. Those share none of their 4-year averaging window, so
    the block stops preserving the dependence it exists to preserve, the bootstrap
    underestimates variance, and the interval comes out too narrow.
    """
    blocks = year_windows(Y, L, full_only=True)
    if not blocks:
        raise ValueError(f"no full {L}-year windows in {int(min(Y))}-{int(max(Y))}")
    J = len(Y)
    pick = rng.integers(len(blocks), size=(B, -(-J // L)))
    return np.stack([np.concatenate([blocks[k] for k in row])[:J] for row in pick])


def bootstrap_rpc(c, n=2000, years=10, drop=3, kind="debiased", seed=0,
                  full_only=True, chunk=16):
    """`n` RPC estimates, each from ONE window of `years` consecutive calendar years
    crossed with ONE independent draw of M-`drop` members without replacement.

    -> (rho_o, rho_m, RPC) each (n, lat, lon), and the list of windows drawn from.

    Unlike a resample of all T cases, every estimate here uses only `years` validation
    points, so each is far noisier and the spread is the aggregate over windows AND
    member draws. The member subset is drawn per estimate, not shared across a replicate.
    """
    F, Gg = np.asarray(c.F, float), np.asarray(c.G, float)
    N, J = F.shape[:2]
    M = N - drop
    if M < 2:
        raise ValueError(f"N={N} leaves M-{drop}={M} members")
    wins = year_windows(c.YEAR, years, full_only)
    if not wins:
        raise ValueError(f"no full {years}-year windows in {c.YEAR.min()}-{c.YEAR.max()}")
    rng = np.random.default_rng(seed)
    Wt = np.zeros((n, J)); Pm = np.zeros((n, N))
    pick = rng.integers(len(wins), size=n)
    for i in range(n):
        k = wins[pick[i]]
        Wt[i, k] = 1.0 / len(k)
        Pm[i, rng.choice(N, M, replace=False)] = 1.0 / M
    o, m, rpc, _ = _eade_boot_at(F, Gg, Wt, Pm, kind, chunk)
    sh = Gg.shape[1:]
    return (o.reshape((n,) + sh), m.reshape((n,) + sh), rpc.reshape((n,) + sh), wins)


def _pci(boot, h, B):
    """Two-sided p from inverting the percentile CI: the level at which it excludes `h`."""
    lo = ((boot <= h).sum(0) + 1) / (B + 1)
    hi = ((boot >= h).sum(0) + 1) / (B + 1)
    return np.clip(2 * np.minimum(lo, hi), 0.0, 1.0)


# ------------------------------------------------------------- resampling schemes

def _perm_idx(J, L, rng, B):
    """Circular block permutation. Each start date used exactly once, so the marginal
    and (bar L-1 joins) the autocorrelation of the permuted series are preserved; only
    the model/obs pairing is destroyed. A random rotation moves the cuts each replicate.
    L=1 is a plain permutation."""
    out = np.empty((B, J), int)
    for b in range(B):
        base = (np.arange(J) + rng.integers(J)) % J
        blocks = [base[i:i + L] for i in range(0, J, L)]
        out[b] = np.concatenate([blocks[k] for k in rng.permutation(len(blocks))])
    return out


def _boot_idx(J, L, rng, B):
    """Circular moving-block bootstrap over array POSITIONS. L=1 is an i.i.d. bootstrap.

    DO NOT use for the Eade path -- positions are not calendar years when the start-year
    set has gaps, and the circular wrap joins the last start date to the first. Use
    `block_idx_years`. Retained only for the i.i.d.-vs-block calibration check in
    test_significance_test_RPC.py, where the synthetic YEAR axis is contiguous.
    """
    st = rng.integers(J, size=(B, -(-J // L), 1))
    return ((st + np.arange(L)) % J).reshape(B, -1)[:, :J]


def _weights(idx, J):
    """(B, J) draw counts / J -- the only thing an estimator sees of a resample."""
    return np.stack([np.bincount(i, minlength=J) for i in idx]).astype(float) / J


def _bh(p, q):
    """Benjamini-Hochberg rejections at FDR level q. Same shape as `p`."""
    flat = np.asarray(p, float).ravel()
    m = flat.size
    o = np.argsort(flat)
    k = np.nonzero(flat[o] <= q * np.arange(1, m + 1) / m)[0]
    out = np.zeros(m, bool)
    if k.size:
        out[o[:k[-1] + 1]] = True
    return out.reshape(np.shape(p))


class Bundle:
    """Minimal stand-in for a handle, for synthetic data. Same attribute contract."""

    def __init__(self, F, G, label="synthetic", units=""):
        self.F = np.asarray(F, float); self.G = np.asarray(G, float)
        self.N, self.J = self.F.shape[:2]
        self.lats = np.zeros(self.F.shape[2]); self.lons = np.arange(self.F.shape[3])
        self.W = np.ones(self.F.shape[2:])
        self.label, self.units = label, units
        self.YEAR = np.arange(self.J); self.MONTH = np.full(self.J, 11)

    @property
    def s(self):
        return self.F.mean(0)

    def gm(self, a):
        m = np.isfinite(a)
        return float((a[m] * self.W[m]).sum() / self.W[m].sum())


# ------------------------------------------------------------------------- the test

class Result:
    """Fields are (lat, lon) unless named `area_*`, `frac_*`, `gm_*` or `*_gm`."""

    def __repr__(self):
        return (f"<Result {self.label} N={self.N} J={self.J} (J_eff~{self.J_eff:.0f}) "
                f"rho_m={self.rho_m_kind} L={self.L} B={self.B}>")

    def report(self, out=sys.stdout):
        p = lambda *a: print(*a, file=out)
        p(repr(self))
        p(f"  point estimate (all {self.N} members)  rho_o {self.gm_rho_o:+.4f}"
          f"   rho_m {self.gm_rho_m:.4f}   RPC {self.RPC_gm:+.3f}")
        p(f"  Eade steps 1-2: T={self.J} cases in blocks of {self.L}, "
          f"M-{self.drop}={self.M} members without replacement, B={self.B}")
        p(f"  step 5, two-tailed {1 - self.alpha:.0%}  (CI does not span the null)")
        p(f"    skill  rho_o != 0 : {self.area_skill:6.1%} of area"
          f"   (of which rho_o>0 {self.area_skill_pos:6.1%})"
          f"   BH-FDR {self.area_skill_fdr:6.1%}")
        p(f"    RPC    != 1       : {self.area_rpc:6.1%} of masked area"
          f"   (RPC>1 {self.area_rpc_gt1:6.1%}, RPC<1 {self.area_rpc_lt1:6.1%})"
          f"   BH-FDR {self.area_rpc_fdr:6.1%}")
        p(f"    global-mean RPC {self.RPC_gm:+.3f}, 5-95% CI "
          f"[{self.RPC_gm_ci[0]:+.3f}, {self.RPC_gm_ci[1]:+.3f}]"
          f"  median {self.RPC_gm_med:+.3f}  p={self.p_RPC_gm:.3f}"
          f"  -> {'REJECT' if self.sig_RPC_gm else 'cannot reject'} RPC=1")
        p(f"  masked area (rho_o>{self.mask_min:g}) {self.area_mask:.1%}"
          f"   frac(RPC>1) {self.frac_gt1:.3f}   [rho_o>0.2: {self.frac_gt1_02:.3f}]")
        p(f"  step-2 effect: median 90% CI width on RPC {self.ci_w:.2f}"
          f" vs {self.ci_w_noM:.2f} without member resampling"
          f"  (x{self.ci_widen:.2f});  pdf median offset {self.med_offset:+.3f}")
        if self.extras:
            p(f"  [not Eade] perfect-model floor: RPC_pm {self.RPC_pm_gm:.3f}"
              f"   frac(RPC>1)|null {self.pm_frac_gt1_mean:.3f}"
              f" ({self.pm_frac_gt1_lo:.3f}-{self.pm_frac_gt1_hi:.3f})")
            p(f"  [not Eade] permutation skill null: {self.area_perm_pw:6.1%} pointwise"
              f"   field-sig p={self.p_field_skill:.4f}")


def test(c, B=1000, L=5, drop=3, alpha=0.10, q=0.10, rho_m="debiased", mask_min=0.0,
         seed=0, extras=False, nom=True, chunk=32, verbose=True):
    """Eade et al. (2014) steps 1-5 on one handle. Two-tailed test at the 90% level.

    B      replicates (Eade: 1000)   L      block length, consecutive years (Eade: 5)
    drop   members held out per replicate, step 2 (Eade: 3)
    alpha  0.10 -> step 5's 5-95% CI      q  BH-FDR level; NOT part of Eade's procedure
    rho_m  "eade" for their published values, "debiased" to remove the 1/N inflation
    extras also run the block-permutation skill null and the perfect-model RPC floor,
           neither of which Eade do; they are reported separately and flagged.
    """
    r = Result()
    F, G, s = np.asarray(c.F, float), np.asarray(c.G, float), np.asarray(c.s, float)
    N, J = F.shape[:2]
    r.label, r.N, r.J, r.L, r.B, r.drop, r.M = c.label, N, J, L, B, drop, N - drop
    r.rho_m_kind, r.alpha, r.q, r.mask_min, r.extras = rho_m, alpha, q, mask_min, extras
    good = (np.isfinite(G).all(0) & np.isfinite(F).all((0, 1))
            & (G.std(0) > 0) & (s.std(0) > 0))
    rng = np.random.default_rng(seed)
    pct = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    sh = s.shape[1:]

    # ---- point estimates, from ALL N members (what gets published)
    r.rho_o = np.where(good, _corr(s, G), np.nan)
    r.rho_m = np.where(good, _rho_m(F, rho_m), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r.RPC = r.rho_o / r.rho_m
    r.gm_rho_o, r.gm_rho_m = c.gm(r.rho_o), c.gm(r.rho_m)
    r.RPC_gm = r.gm_rho_o / r.gm_rho_m
    r.mask = good & (r.rho_o > mask_min)
    r.area_mask = _afrac(c, r.mask, good)
    r.frac_gt1 = _afrac(c, r.RPC > 1, r.mask)
    r.frac_gt1_02 = _afrac(c, r.RPC > 1, good & (r.rho_o > 0.2))
    r.J_eff = _jeff(c, np.where(good, G, np.nan))

    # ---- Eade steps 1-4. Wt and Pm are built here rather than inside `_eade_boot` so
    # the no-step-2 comparison below can reuse the SAME start-date resample: paired, so
    # the width ratio measures step 2 and not the difference between two RNG streams.
    Wt = _weights(block_idx_years(c.YEAR, L, rng, B), J)               # step 1
    Pm = np.zeros((B, N))
    for b in range(B):                                                 # step 2
        Pm[b, rng.choice(N, N - drop, replace=False)] = 1.0 / (N - drop)
    bo, bm, brpc, brgm = _eade_boot_at(F, G, Wt, Pm, rho_m, chunk,
                                       c.W.ravel(), good.ravel())
    bo = bo.reshape((-1,) + sh); bm = bm.reshape((-1,) + sh)
    brpc = brpc.reshape((-1,) + sh)

    # ---- step 5: reject at the 90% level if the 5-95% CI does not span the null value
    r.rho_o_ci = _nan(np.nanpercentile, bo, pct, axis=0)
    r.rho_m_ci = _nan(np.nanpercentile, bm, pct, axis=0)
    r.RPC_ci = _nan(np.nanpercentile, brpc, pct, axis=0)
    r.RPC_med = _nan(np.nanmedian, brpc, axis=0)
    r.sig_skill = np.where(good, (r.rho_o_ci[0] > 0) | (r.rho_o_ci[1] < 0), False)
    r.sig_skill_pos = np.where(good, r.rho_o_ci[0] > 0, False)
    r.sig_rpc = np.where(r.mask, (r.RPC_ci[0] > 1) | (r.RPC_ci[1] < 1), False)
    r.sig_rpc_gt1 = np.where(r.mask, r.RPC_ci[0] > 1, False)
    r.sig_rpc_lt1 = np.where(r.mask, r.RPC_ci[1] < 1, False)
    for nm, m_ in (("skill", good), ("skill_pos", good), ("rpc", r.mask),
                   ("rpc_gt1", r.mask), ("rpc_lt1", r.mask)):
        setattr(r, f"area_{nm}", _afrac(c, getattr(r, f"sig_{nm}"), m_))
    # p-values from the SAME pdf, so the project's FDR view stays available. Eade apply
    # no correction, so this is an addition, never a replacement for the areas above.
    r.p_skill = np.where(good, _pci(bo, 0.0, B), np.nan)
    r.p_rpc = np.where(good, _pci(brpc, 1.0, B), np.nan)
    r.sig_skill_fdr = _fdr_on(r.p_skill, good, q)
    r.sig_rpc_fdr = _fdr_on(r.p_rpc, r.mask, q)
    r.area_skill_fdr = _afrac(c, r.sig_skill_fdr, good)
    r.area_rpc_fdr = _afrac(c, r.sig_rpc_fdr, r.mask)
    # the global-mean RPC: ONE test per configuration, so no multiple-testing problem
    r.RPC_gm_ci = np.nanpercentile(brgm, pct)
    r.RPC_gm_med = float(np.nanmedian(brgm))
    r.p_RPC_gm = float(_pci(brgm[:, None], 1.0, B)[0])
    r.sig_RPC_gm = bool((r.RPC_gm_ci[0] > 1) or (r.RPC_gm_ci[1] < 1))

    # ---- how much step 2 actually matters: same start dates, all N members.
    # Chunked: `_boot_stats` on all B rows at once allocates (B, N, cells) several times
    # over, which is 2.8 GB at N=66 B=1000 and was enough to OOM a 16 GB allocation.
    b1 = np.empty((B,) + sh) if nom else None
    for i in range(0, B, chunk) if nom else ():
        o1, m1 = _boot_stats(F, G, Wt[i:i + chunk], rho_m)
        with np.errstate(invalid="ignore", divide="ignore"):
            b1[i:i + chunk] = (o1 / m1).reshape((-1,) + sh)
    with np.errstate(invalid="ignore", divide="ignore"):
        r.RPC_ci_noM = (_nan(np.nanpercentile, b1, pct, axis=0) if nom
                        else np.full_like(r.RPC_ci, np.nan))
    with np.errstate(invalid="ignore"):
        r.ci_w = float(_nan(np.nanmedian, (r.RPC_ci[1] - r.RPC_ci[0])[r.mask]))
        r.ci_w_noM = float(_nan(np.nanmedian,
                                (r.RPC_ci_noM[1] - r.RPC_ci_noM[0])[r.mask]))
        r.ci_widen = r.ci_w / r.ci_w_noM
        r.med_offset = float(_nan(np.nanmedian, (r.RPC_med - r.RPC)[r.mask]))

    # ---- additions, clearly not part of Eade's five steps
    if extras:
        r.RPC_pm, r.pm_fracs = _perfect_model(c, F, rho_m, good, mask_min)
        r.RPC_pm_gm = c.gm(r.RPC_pm)
        r.pm_frac_gt1_mean = float(_nan(np.nanmean, r.pm_fracs))
        r.pm_frac_gt1_lo, r.pm_frac_gt1_hi = _nan(np.nanpercentile, r.pm_fracs, [5, 95])
        sc = (s - s.mean(0)) / s.std(0)
        Gc = (G - G.mean(0)) / np.where(good, G.std(0), np.nan)
        null = np.empty((B,) + sh)
        idx = _perm_idx(J, L, rng, B)
        for b in range(B):
            null[b] = (sc * Gc[idx[b]]).mean(0)
        r.p_perm = (1 + (null >= r.rho_o).sum(0)) / (B + 1)
        r.sig_perm_pw = np.where(good, r.p_perm < alpha / 2, False)
        r.area_perm_pw = _afrac(c, r.sig_perm_pw, good)
        crit = _nan(np.nanquantile, null, 1 - alpha / 2, axis=0)
        a_null = np.array([_afrac(c, null[b] > crit, good) for b in range(B)])
        r.p_field_skill = (1 + (a_null >= r.area_perm_pw).sum()) / (B + 1)
    if verbose:
        r.report()
    return r


def _perfect_model(c, F, kind, good, mask_min):
    """Hold each member out as pseudo-obs; RPC from the other N-1. -> (RPC_pm, fracs).

    corr(s_-n, f_n) = p^2/sqrt(p^2+(1-p^2)/(N-1)) is the same function of ensemble size
    as corr(s_N, G) = p^2/sqrt(p^2+(1-p^2)/N), so this is the real statistic at N-1, and
    the debiased rho_m is near-flat in N. For a POOLED handle the held-out member also
    carries structural error, which is the multi-model analogue and is deliberate.
    """
    N = F.shape[0]
    Fsum, v, sq = F.sum(0), F.var(axis=1), None
    if kind == "eade":
        sq = np.sqrt(v).sum(0)
    vsum, out, fr = v.sum(0), [], []
    for n in range(N):
        loo = (Fsum - F[n]) / (N - 1)
        tot2 = (vsum - v[n]) / (N - 1)
        vs = loo.var(0)
        rm = (np.sqrt(vs) / ((sq - np.sqrt(v[n])) / (N - 1)) if kind == "eade" else
              np.sqrt(np.maximum(((N - 1) * vs - tot2) / (N - 2), 0.0) / tot2))
        ro = _corr(loo, F[n])
        with np.errstate(invalid="ignore", divide="ignore"):
            rpc = np.where(good, ro / rm, np.nan)
        out.append(rpc)
        fr.append(_afrac(c, rpc > 1, good & (ro > mask_min)))
    return _nan(np.nanmedian, out, axis=0), np.array(fr)


def _fdr_on(p, m, q):
    """BH-FDR at level q over the cells in `m` only; everything else False."""
    out = np.zeros(np.shape(p), bool)
    sel = np.asarray(m, bool) & np.isfinite(p)
    if sel.any():
        out[sel] = _bh(np.asarray(p)[sel], q)
    return out


def _afrac(c, ind, m):
    """Area-weighted fraction of `ind` over the cells in `m`."""
    if not np.any(m):
        return float("nan")
    return c.gm(np.where(m, np.asarray(ind, float), np.nan))


def _jeff(c, G):
    """J (1-r1)/(1+r1) from the observations' lag-1 autocorrelation -- a diagnostic."""
    a = G - np.nanmean(G, 0)
    r1 = np.nansum(a[1:] * a[:-1], 0) / np.nansum(a * a, 0)
    r = float(np.clip(c.gm(r1), 0, 0.95))
    return G.shape[0] * (1 - r) / (1 + r)
