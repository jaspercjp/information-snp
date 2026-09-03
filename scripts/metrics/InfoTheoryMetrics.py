import os as _os, sys as _sys  # noqa: E401  -- snp_path bootstrap, see scripts/snp_path.py
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import snp_path as _snp_path  # noqa: E402,F401  -- all scripts/ subfolders onto sys.path

from scipy.special import gammaln
import numpy as np
from tqdm import tqdm
import xarray as xr
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import iqr 

def freedman_diaconis_bins(X):
    return int(np.ptp(X) / (2 * iqr(X) / np.power(len(X), 1/3)))

# Knuth(2006) Data-Based Binning
def optbins(data,maxM):
    """ Assumes one-dimensional data, and the probabilities are NOT normalized!"""
    N = len(data)
    
    # Simply loop through the different numbers of bins
    # and compute the posterior probability for each.
    logp = np.zeros(maxM)
    for M in range(1,maxM):
        n, _ = np.histogram(data,bins=M) # Bin the data (equal width bins here)
        # print(n)
        part1 = N*np.log(M) + gammaln(M/2) - gammaln(N+M/2)
        part2 = -M*gammaln(1/2) + np.sum(gammaln(n+0.5))
        logp[M] = part1 + part2
    
    optM = np.argmax(logp) + 1
    return optM

def prob_normalize(hist_counts):
    return hist_counts.astype(np.float32) / np.sum(hist_counts)

def H(X, bins=None):
    """ Uses histograms to estimate entropy of an array. """
    N = len(X)
    if bins is None:
        X_pdf, edges = np.histogram(X, bins=int(np.sqrt(N/5)), density=True)
    else:
        X_pdf, edges = np.histogram(X, bins=bins, density=True)
    # p_i = prob_normalize(X_hist_counts)
    bin_width = edges[1]-edges[0]
    p_i = X_pdf * bin_width
    p_i = p_i[np.where(p_i > 0)]
    return -np.sum(p_i * np.log2(p_i))

def I(X,Y, nbins_alt=None):
    """ Uses histograms to estimate the mutual information between two arrays of equal lengths. """
    N = len(X)
    if nbins_alt is None:
        XY_counts, _, _ = np.histogram2d(X,Y, bins=int(np.sqrt(N/5)))
    else:
        XY_counts, _, _ = np.histogram2d(X,Y, bins=nbins_alt)
    p_XY = prob_normalize(XY_counts)
    p_X = np.sum(p_XY, axis=1)
    p_Y = np.sum(p_XY, axis=0)
    p_XY = p_XY[np.where(p_XY > 0)]
    p_X = p_X[np.where(p_X>0)]
    p_Y = p_Y[np.where(p_Y>0)]
    H_X = -np.sum(p_X * np.log2(p_X))
    H_Y = -np.sum(p_Y * np.log2(p_Y))
    H_XY = -np.sum(p_XY * np.log2(p_XY))
    return H_X + H_Y - H_XY, H_X, H_Y

def I_jugaad(obs, model_output, nbins_alt=None):
    """ Uses the flattening trick described in the paper to calculate mutual information.
        obs: observation data of length L
        model_output: ensemble model output of size N x L
    """
    N,l = model_output.shape
    obs_jugaad = np.tile(obs, N)
    if type(model_output) is np.ndarray:
        return I(obs_jugaad, model_output.reshape(N*l), nbins_alt=nbins_alt)
    else:
        return I(obs_jugaad, model_output.to_numpy().reshape(N*l), nbins_alt=nbins_alt)
    
def TC(X):
    """ Calculates total correlation (Not used) """
    N,m = X.shape
    tc = 0
    entropies = np.zeros(N)
    for n in range(N-1):
        MI, H_xn, H_X1n = I_jugaad(X[n+1,:], X[0:n+1,:])
        tc += MI 
        entropies[n] = H_xn
    return tc, entropies

def info_RPC(ensemble:xr.Dataset, signal:xr.Dataset, observation:xr.Dataset, 
             method='histogram', quant_uncert=True):
    """
        Takes in an ensemble model, a signal, and an observation of appropriate sizes 
        (signal and observation should have the same length L, while the ensemble should be NxL where N is 
        the number of members) and estimates the RPC measures based on information theory.
    """
    N, _, x_len, y_len = ensemble.psl.shape
    if quant_uncert and method=='histogram':
        alts = [0.5, 1, 1.5] # For uncertainty quantification
    else:
        alts = [1]
    I_gf_mat = np.zeros((len(alts), x_len, y_len))
    I_sf_mat = np.zeros((len(alts), x_len, y_len))
    H_f_mat = np.zeros((len(alts), x_len, y_len))
    H_g_mat = np.zeros((len(alts), x_len, y_len))
    for lat in tqdm(range(x_len), desc="Calculating Entropy Metrics"):
        for lon in range(y_len):
                g_t = observation.psl[:,lat,lon].to_numpy()
                s_t = signal.psl[:,lat,lon].to_numpy()
                f_t = ensemble.psl[:,:,lat,lon].to_numpy()
                # Replace nan with 0's for calculations
                f_t[np.where(np.isnan(f_t))] = 0.0
                s_t_ext = np.tile(s_t, len(ensemble.n))
                g_t_ext = np.tile(g_t, len(ensemble.n))
                f_t_flat = f_t.flatten()
                
                if method=='histogram':
                    # Use the Freedman-Diaconis rule to bin the data 
                    f_bins = freedman_diaconis_bins(f_t_flat)
                    s_bins = freedman_diaconis_bins(s_t)
                    s_bins_ext = freedman_diaconis_bins(s_t_ext) 
                    g_bins_ext = freedman_diaconis_bins(g_t_ext)
                    g_bins = freedman_diaconis_bins(g_t)
                elif method!='kraskov':
                    # Kraskov estimator was not used in the paper, but is kept here for potential future use
                    raise Exception(f"Entropy estimation method {method} not recognized.")
                
                for k in range(len(alts)):
                    if method=='histogram':
                        I_gf, H_g, _ = I_jugaad(g_t, f_t, 
                                                nbins_alt=(int(np.ceil(g_bins_ext*alts[k])), 
                                                           int(np.ceil(f_bins*alts[k]))))
                        I_sf, _, H_f = I_jugaad(s_t, f_t, 
                                                nbins_alt=(int(np.ceil(s_bins_ext*alts[k])), 
                                                           int(np.ceil(f_bins*alts[k]))))
                    elif method=='kraskov':
                        I_fg = mutual_info_regression(f_t_flat.T.reshape(-1,1), g_t_ext)[0]
                        I_sf = mutual_info_regression(f_t_flat.T.reshape(-1,1), s_t_ext)[0]
                        H_f = mutual_info_regression(f_t_flat.T.reshape(-1,1), f_t_flat)[0]
                        H_g = mutual_info_regression(g_t.T.reshape(-1,1), g_t)[0]
                    I_gf_mat[k,lat,lon] = I_gf
                    I_sf_mat[k,lat,lon] = I_sf
                    H_f_mat[k,lat,lon] = H_f
                    H_g_mat[k,lat,lon] = H_g
    
    return I_gf_mat, I_sf_mat, H_f_mat, H_g_mat