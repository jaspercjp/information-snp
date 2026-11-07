ExchangeabilityMaps: are the ensemble members exchangeable?
===========================================================

Two slides, same layout and colour scale: columns SLP, PRECT, TREFHT; colour is the
excess I(M; X, Y) (observed minus its permutation-null mean, bits) where p <= 0.05,
flat grey elsewhere. M is the member label, X the mean of the other members, Y the
member. Under exchangeable members about 5% of the area rejects by chance.

  exchangeability-all-models  all members pooled: members are not exchangeable
  exchangeability-canesm5     CanESM5 alone: its members compared with each other

Rows
  DJF          one value per start date. All models: lead months 2-4, the five
               November models (CanESM5 starts in January and is not in that pool).
               CanESM5: lead months 12-14, its first full DJF (one year longer lead).
  Decade       the monthly decade s1961 (PRECT s1978), each model's month-of-year
               climatology removed: one continuous 120-month run per member.

Test: I(M; X, Y) with a member-label permutation null, 200 permutations, the per-member
detrend done inside the test (on raw members, so every permuted member is detrended
like the observed ones). DJF: labels permuted per start date (exact). Decade: in
12-month blocks (validated for lag-1 autocorrelation up to about 0.5; above that, as in
tropical TREFHT, the null over-rejects). Field significance: the permutations' own
area shares (spatial correlation kept).

Rejecting area (%), observed | permutation null 5-95% | field p

  all members  DJF, lead months 2-4             SLP      26.5 |  3.2- 6.9 | 0.005   R=62, T=51
  all members  DJF, lead months 2-4             PRECT    43.4 |  4.0- 5.9 | 0.005   R=62, T=33
  all members  DJF, lead months 2-4             TREFHT   36.6 |  3.4- 7.2 | 0.005   R=62, T=52
  all members  Decade, seasonal cycle removed   SLP      42.3 |  3.5- 6.8 | 0.005   R=92, T=120
  all members  Decade, seasonal cycle removed   PRECT    74.8 |  4.0- 6.1 | 0.005   R=112, T=120
  all members  Decade, seasonal cycle removed   TREFHT   43.8 |  3.2- 7.9 | 0.005   R=112, T=120
  CanESM5      DJF, lead months 12-14           SLP       4.0 |  3.7- 6.5 | 0.861   R=20, T=58
  CanESM5      DJF, lead months 12-14           PRECT     4.1 |  4.3- 5.7 | 0.980   R=40, T=42
  CanESM5      DJF, lead months 12-14           TREFHT    4.8 |  4.0- 6.1 | 0.587   R=40, T=60
  CanESM5      Decade, seasonal cycle removed   SLP       9.7 |  3.5- 6.5 | 0.005   R=20, T=120
  CanESM5      Decade, seasonal cycle removed   PRECT     6.4 |  4.2- 6.2 | 0.030   R=40, T=120
  CanESM5      Decade, seasonal cycle removed   TREFHT   23.2 |  3.8- 6.5 | 0.005   R=40, T=120

Regenerate (compute node, repository root):
  python scripts/exchangeability_check.py grid --case VAR:CASE [--model CanESM5] --block {1,12} --redetrend
  python scripts/exchangeability_check.py figure-slides
  python scripts/figure_set.py Presentation refresh && ... verify
Inputs: $SNP_OUT/exchangeability_grid_*_redetrend.npz; per-case JSON in figures/data.
