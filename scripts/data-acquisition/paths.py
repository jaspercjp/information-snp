"""Where everything in data-acquisition/ reads and writes. The one place the data root is set.

    DATA = $SNP_DATA, else <repo>/data

Raw DCPP downloads (215 GB) live outside the repo, in $SCRATCH/dcpp by default; set
$SNP_DCPP_RAW to point elsewhere. $SCRATCH purges files unmodified for 90 days, so copy
anything that must outlive a quarter to Oak.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.environ.get("SNP_DATA", os.path.join(REPO, "data"))

OBS = os.path.join(DATA, "obs")                            # verifying observations
HADSLP2 = os.path.join(OBS, "hadslp2")                     # thesis obs (HadSLP2, to 2004)
MODELS_CMIP5 = os.path.join(DATA, "models")                # CMIP5 decadal1961 psl
ENSEMBLES = os.path.join(DATA, "ensembles")                # the thesis ensemble
DCPP_MANIFEST = os.path.join(DATA, "dcpp_files.json")
DCPP_RAW = os.environ.get("SNP_DCPP_RAW",
                          os.path.join(os.environ.get("SCRATCH", DATA), "dcpp"))
DCPP_CUBES = os.path.join(DATA, "dcpp_cubes")              # <MODEL>/<VAR>_lead<a>-<b>.npz
DEPRESYS_MANIFEST = os.path.join(DATA, "depresys4_files.json")
DEPRESYS_RAW = os.path.join(DATA, "depresys4")
DEPRESYS_CUBES = os.path.join(DATA, "depresys4_cubes")
MONTHLY_CACHE = os.path.join(DATA, "dcpp_monthly_cache")   # <MODEL>/<VAR>_s<YYYY>_...npz
CASES = os.path.join(DATA, "cases")                        # make_cases.py output
