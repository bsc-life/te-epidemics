__author__ = "Miguel Ponce de León"
__email__  = "miguel.ponce@bsc.es"

import numpy as np
import xarray as xr
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler


def derivates_embedding(ts, normalize=False, eps=1e-4):
    L = ts.shape[0]
    if normalize:
        scaler = MinMaxScaler()
        ts = ts.values.reshape((L, 1))
        ts = scaler.fit_transform(ts)
        ts = pd.Series(ts.reshape(L))
    
    deltas = ts.diff().fillna(0)
    symbols = np.zeros(deltas.shape)
    symbols[deltas > eps] = 1
    symbols[deltas < -eps] = 2
    return symbols

def symbolic_embedding(ts, m=3):
    symbol_list = list(permutations(range(m)))
    symbols_dict = {s:i for i,s in enumerate(symbol_list)}
    L = ts.shape[0]
    ts_s = np.zeros(L)
    for i in range(L-m+1):
        w = tuple(np.argsort(ts[i:i+m]))
        ts_s[i] = symbols_dict[w]

    ts_s[L-m:] = ts_s[L-m]
    return np.array(ts_s)

def uniform_embedding(ts, n_bins=20):
    L = ts.shape[0]
    idx = ts.index
    enc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    ts = ts.values.reshape((L, 1))
    ts_hat = enc.fit_transform(ts)
    ts_hat = ts_hat.reshape(L)
    return pd.Series(ts_hat, index=idx)


def symmetric_uniform_embedding(ts, local_max, n_bins=20):
    L = ts.shape[0]
    ts_hat = np.zeros(L)
    w = local_max/n_bins
    for i in range(n_bins):
        ts_hat[ts>w*i] = i+1      
    ts_hat = pd.Series(ts_hat, index=ts.index)
    return ts_hat
        
   