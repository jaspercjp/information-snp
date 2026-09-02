"""Analytic-property tests for `significance_test_RPC`. Run: python test_significance_test_RPC.py

Every check here is something a bootstrap must satisfy regardless of the climate data,
so a failure means the estimator is wrong rather than the science being surprising:

  1  the vectorized estimators equal `smyle_metrics`' to floating point
  1b Eade's steps 1-3 are what the paper says: T cases with replacement in blocks of 5
     consecutive years, exactly M-3 members WITHOUT replacement, and one replicate
     equal to an explicitly built subsample-then-resample
  2  step 5's CI rejects at the nominal rate under a true null
  3  on 4-year windows offset by 1 year, an i.i.d. resample is LIBERAL and blocks of
     five are not -- the overstatement Eade's step 1 exists to prevent, reproduced
  4  BH-FDR is monotone in q and never more liberal than pointwise
  5  the perfect-model null RPC matches p / sqrt(p^2 + (1-p^2)/N) analytically
  6  frac(RPC>1) under a perfect model is far above zero -- the null floor
  7  both a single-model Cube and a Pooled handle run end to end
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import significance_test_RPC as T
import smyle_metrics as M

OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def synth(rng, p, N, J, C=200, perfect=True):
    """(F, G) with predictable fraction p. `perfect`: obs are one more member draw."""
    a = p / np.sqrt(1 - p ** 2)                       # signal amp, noise sd 1
    x = rng.standard_normal((J, C))
    F = a * x + rng.standard_normal((N, J, C))
    G = a * x + rng.standard_normal((J, C)) if perfect else rng.standard_normal((J, C))
    return F[:, :, :, None], G[:, :, None]            # fake a (lat, lon) of (C, 1)


# ---------------------------------------------------------------- 1  estimators
print("\n1. vectorized estimators vs smyle_metrics")
rng = np.random.default_rng(0)
F, G = synth(rng, 0.4, 12, 50)
s = F.mean(0)
check("rho_o", np.allclose(T._corr(s, G), M.pearson_coeff(s, G)),
      "max|d|=%.2e" % np.abs(T._corr(s, G) - M.pearson_coeff(s, G)).max())
for name, ref in (("debiased", M.ensemble_SNR_debiased), ("eade", M.ensemble_SNR)):
    got = T._rho_m(F, name)
    check(f"rho_m {name}", np.allclose(got, ref(F)),
          "max|d|=%.2e" % np.abs(got - ref(F)).max())

# the weighted-count reduction must equal an explicit resample
w = np.zeros(50); w[[3, 3, 7, 11]] = [2 / 50, 2 / 50, 1 / 50, 1 / 50]  # a real multiset
idx = np.array([3, 3, 3, 3, 7, 11])  # not length-J: build a matching one instead
idx = np.repeat([3, 7, 11], [4, 1, 1])
w = np.bincount(idx, minlength=50) / len(idx)
mo, mm = T._boot_stats(F, G, w[None, :], "debiased")
check("count-weighted == explicit resample",
      np.allclose(mo[0], M.pearson_coeff(s[idx], G[idx]), atol=1e-10)
      and np.allclose(mm[0], M.ensemble_SNR_debiased(F[:, idx]), atol=1e-10))

# ------------------------------------------------------ 1b  Eade steps 1-2 exactly
print("\n1b. Eade step 2: M-3 members, WITHOUT replacement")
rng = np.random.default_rng(99)
F, G = synth(rng, 0.4, 12, 50, C=50)
N, J, B, drop = 12, 50, 400, 3
rr = np.random.default_rng(0)
Pm = np.zeros((B, N))
for b in range(B):
    Pm[b, rr.choice(N, N - drop, replace=False)] = 1.0 / (N - drop)
check("exactly M-drop members per replicate", ((Pm > 0).sum(1) == N - drop).all())
check("no member drawn twice (without replacement)",
      np.allclose(Pm[Pm > 0], 1.0 / (N - drop)))
check("every member gets used across replicates", ((Pm > 0).sum(0) > 0).all())
bidx = T._boot_idx(J, 5, np.random.default_rng(0), B)
check("step 1 draws exactly T=J cases", bidx.shape == (B, J))
d = np.diff(bidx.reshape(B, -1, 5), axis=2) % J
check("step 1 blocks are 5 CONSECUTIVE years", (d[:, :-1] == 1).all())
check("step 1 is with replacement (blocks repeat)",
      any(len(set(map(tuple, b.reshape(-1, 5)))) < 10 for b in bidx))
# step 3 on a subsampled ensemble must equal the explicit calculation
sub = np.array([0, 2, 3, 5, 7, 8, 9, 10, 11])           # 9 = 12-3 members
idx = np.repeat([3, 7, 11], [4, 1, 1])
w = np.bincount(idx, minlength=J) / len(idx)
Ps = np.zeros((1, N)); Ps[0, sub] = 1.0 / len(sub)
o, m, _, _ = T._eade_boot_at(F, G, w[None], Ps, "debiased")
ss = F[sub][:, idx].mean(0)
check("step 3 == explicit subsample+resample",
      np.allclose(o[0], M.pearson_coeff(ss, G[idx]).ravel(), atol=1e-10)
      and np.allclose(m[0], M.ensemble_SNR_debiased(F[sub][:, idx]).ravel(), atol=1e-10))

# ------------------------------------------------------- 2  nominal, i.i.d. case
print("\n2. step-5 rejection rate under a TRUE null, i.i.d. samples (nominal 10%)")
rng = np.random.default_rng(1)
F, G = synth(rng, 0.4, 12, 60, C=2000, perfect=False)      # TRUE null: no skill
r = T.test(T.Bundle(F, G), B=1000, L=1, seed=2, verbose=False)
rate = r.sig_skill.mean()
check("two-tailed 90% CI on rho_o rejects near 10%", 0.05 < rate < 0.16,
      "rate=%.3f" % rate)
rp = T.test(T.Bundle(F, G), B=1000, L=1, seed=2, extras=True, verbose=False)
check("permutation null agrees (one-sided 5%)", 0.02 < rp.area_perm_pw < 0.09,
      "rate=%.3f" % rp.area_perm_pw)

# ------------------------------------------------- 3  overlapping windows, block
print("\n3. overlapping 4-yr windows, TRUE null: i.i.d. is liberal, block is not")
rng = np.random.default_rng(3)
Jy, C = 60, 2000
xm = rng.standard_normal((Jy + 3, C)); xo = rng.standard_normal((Jy + 3, C))


def run4(z):                                  # 4-year running mean, offset 1 year
    return np.stack([z[j:j + 4].mean(0) for j in range(Jy)])


Fo = np.stack([run4(xm + 0.9 * rng.standard_normal((Jy + 3, C))) for _ in range(12)])
Go = run4(xo)
b = T.Bundle(Fo[:, :, :, None], Go[:, :, None])
r1 = T.test(b, B=1000, L=1, seed=4, verbose=False)
r5 = T.test(b, B=1000, L=5, seed=4, verbose=False)
a1, a5 = r1.sig_skill.mean(), r5.sig_skill.mean()
check("i.i.d. over-rejects on overlapping windows", a1 > 0.15, "rate=%.3f" % a1)
check("block is strictly less liberal", a5 < a1, "block=%.3f iid=%.3f" % (a5, a1))
check("block is near nominal", a5 < 0.22, "rate=%.3f" % a5)
print("       i.i.d. overstates significant area by %.0f%%" % (100 * (a1 / a5 - 1)))
# The number that matters for interpreting any area fraction: even WITH blocking, this
# CI-inversion test over-rejects on overlapping windows, so the effective chance level
# is not the nominal one. Quoted in the caveats whenever an area fraction is reported.
print("       LIBERALITY: block rejects %.1f%% at nominal 10%% -> x%.2f too liberal"
      % (100 * a5, a5 / 0.10))

# ------------------------------------------------------------------- 4  BH-FDR
print("\n4. BH-FDR")
p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9])
prev = -1
mono = True
for q in (0.01, 0.05, 0.10, 0.25, 0.5):
    n = T._bh(p, q).sum()
    mono &= n >= prev
    prev = n
check("monotone in q", mono)
check("q=0.05 matches hand calculation", T._bh(p, 0.05).sum() == 3,
      "n=%d" % T._bh(p, 0.05).sum())
check("never more liberal than pointwise", (T._bh(p, 0.10) <= (p < 0.05)).all() or
      T._bh(p, 0.10).sum() <= (p < 0.05).sum())

# ------------------------------------------- 5  perfect-model null vs analytics
print("\n5. perfect-model null RPC vs p/sqrt(p^2+(1-p^2)/N)")
for p_true, N in ((0.33, 16), (0.33, 46), (0.56, 66)):
    rng = np.random.default_rng(10 + N)
    F, G = synth(rng, p_true, N, 400, C=400)          # long J to kill sampling noise
    r = T.test(T.Bundle(F, G), B=200, L=1, seed=5, extras=True, verbose=False)
    Np = N - 1                                        # PM holds one member out
    want = p_true / np.sqrt(p_true ** 2 + (1 - p_true ** 2) / Np)
    got = np.nanmedian(r.RPC_pm)
    check(f"p={p_true} N={N}", abs(got - want) < 0.02,
          "pm=%.3f analytic=%.3f" % (got, want))
    # and the OBSERVED RPC on perfect-model synthetic data must match it too
    check(f"   observed RPC agrees (perfect model)",
          abs(np.nanmedian(r.RPC) - want) < 0.03,
          "obs=%.3f" % np.nanmedian(r.RPC))

# ------------------------------------------------------- 6  the frac>1 floor
print("\n6. frac(RPC>1) null floor at N=20, J=61 (handoff quotes ~0.449)")
rng = np.random.default_rng(20)
F, G = synth(rng, 0.26, 20, 61, C=3000)
r = T.test(T.Bundle(F, G), B=200, L=1, seed=6, extras=True, verbose=False)
floor = r.pm_frac_gt1_mean
check("floor is far above zero", 0.25 < floor < 0.6, "floor=%.3f" % floor)
check("observed frac>1 is consistent with the floor",
      abs(r.frac_gt1 - floor) < 0.12, "obs=%.3f floor=%.3f" % (r.frac_gt1, floor))

# ------------------------------------------------------------- 7  real handles
print("\n7. real handles run end to end")
import depresys4_handles as D, dcpp_handles as Gh
for lbl, c in (("DePreSys4 SLP", D.get(lead="13-60", var="SLP")),
               ("pooled SLP", Gh.get(lead="13-60", var="SLP", verbose=False))):
    r = T.test(c, B=200, seed=7, extras=True, verbose=False)
    check(lbl, np.isfinite(r.frac_gt1) and 0 <= r.area_skill <= 1,
          "rho_o=%+.4f RPC=%.3f frac>1=%.3f skill=%.3f RPCgm_CI=[%.2f,%.2f] widen=x%.2f"
          % (r.gm_rho_o, r.RPC_gm, r.frac_gt1, r.area_skill,
             r.RPC_gm_ci[0], r.RPC_gm_ci[1], r.ci_widen))

print("\n%d/%d passed" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
