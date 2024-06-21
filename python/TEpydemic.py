import numpy as np
import xarray as xr

from itertools import product, permutations
from pyinform import transfer_entropy
from multiprocessing import Pool


class Embedding(object):
    def __init__(self, data) -> None:
        self._data = data
        pass

    def transform(self):
        return self._data
    

def _calc_te(args):
    X, Y, omega, delta, s, k = args
    N = X.shape[0] - (omega + delta) + 1
    n_points = int( N / s )
    TE = np.zeros(n_points)
    # Speed-up to avoid calculating cases where there is no risk between i and j
    if X.max() == 0:
        return TE

    for j,i in enumerate(range(0, N, s)):
        TE[j] = transfer_entropy( X[i:i+omega], 
                                  Y[i+delta:i+delta+omega], k)
    return TE


class TEpydemic(object):
    """
    Initialize a global model object for multiprocessing.
    
    Parameters
    ----------
    movements: xarray.DataArray 
        The 2-dimensional time serie of dims (source, target, time) 
        that stores daily flow matrix (flows can be mobility or risk)
    
    cases: xarray.DataArray 
        The 1-dimensional time serie of dims (source, target, time) where
        target is the the list of mobility areas

    omega: int
        Size of the sliding window

    delta: int
        Time offset between X and Y use when measuring
        the Transfer entropy between
    
    s: int
        Step used to move the sliding window

    embedding: string
        The name of the approch used to make the data
        a symbolic time serie
                    
    """
    def __init__(self, movement, cases, 
                 omega=5, delta=2, s=1,
                 embedding="round", k=1,
                 n_bins=20, symmetric_binning=True,
                 use_case_as_x=False
                 ) -> None:

        self._movement = movement
        self._movement_symbolic = None
        self._cases = cases
        self._cases_symbolic = None

        self._omega = omega
        self._delta = delta
        self._s = s
        self._k = k
        self._embedding = embedding
        
        self._symmetric_binning = symmetric_binning
        self._n_bins = n_bins
        
        self._use_case_as_x = use_case_as_x
        
        self._check_params()

        self._date = self._cases.coords['date']
        self._patches = self._cases.coords['id']
        
        self._econde_data()
    
    def _check_params(self):

        assert type(self._movement) == xr.core.dataarray.DataArray
        assert type(self._cases) == xr.core.dataarray.DataArray

        assert self._movement.dims == ("source", "target", "date")
        assert self._cases.dims == ("id", "date")

        assert (self._movement.coords["source"].values == 
                self._movement.coords["target"].values ).all()

        assert (self._movement.coords["source"].values ==
                self._cases.coords["id"].values ).all()

        assert (self._movement.coords["date"].values ==
                self._cases.coords["date"].values).all()
        
        assert self._embedding in ("round", "uniform")
        
    def _econde_data(self):
        if self._embedding == "round":
            self._movement_symbolic = self._movement.copy()
            self._movement_symbolic = self._movement.round()
            self._cases_symbolic = self._cases.copy()
            self._cases_symbolic = self._cases_symbolic.round()
            
            
        if self._embedding == "uniform":
            self._movement_symbolic = self._movement.copy()
            XMax = self._movement_symbolic.max('date').values
            if self._symmetric_binning:
                XMax[XMax-XMax.T < 0] = 0
                XMax += XMax.T
            np.fill_diagonal(XMax, 1)

            scale = 100
            self._movement_symbolic = self._movement_symbolic.transpose('date', 'source', 'target') / XMax
            self._movement_symbolic *= scale
            self._movement_symbolic = self._movement_symbolic.transpose('source', 'target', 'date')
            bins = np.linspace(0, scale,  self._n_bins,)
            my_digitize = lambda x,bins: np.digitize(x, bins, right=False)
            self._movement_symbolic = xr.apply_ufunc(my_digitize, self._movement_symbolic, bins)
            
            self._cases_symbolic = self._cases.copy()

            YMax = self._cases_symbolic.max('date').values
          
    def calculate_te(self, cpus=1):
        results = None
        if cpus > 1:
            with Pool(cpus) as pool:
                X = self._movement_symbolic
                Y = self._cases_symbolic
                omega = self._omega
                delta = self._delta
                s = self._s
                k = self._k
                patches_ids = self._patches
                results = pool.map(_calc_te,
                                    [(X.loc[i,j,:], Y.loc[j,:], omega, delta, s, k) 
                                        for i,j in product(patches_ids, repeat=2)])
        else:
            for i,j in product(patches_ids, repeat=2):
                TE = _calc_te(X.loc[i,j,:], Y.loc[j,:], omega, delta, s, k)

        return results


if __name__ == "main":
    cases = xr.load_dataset('../experiments/TE_cnig_provincias_no_embedding/cases_pop_ds.nc')['new_cases_by_100k']
    risk_ij = xr.load_dataset('../experiments/TE_cnig_provincias_no_embedding/risk_ds.nc')['risk_ij']
    risk_hat_ij = xr.load_dataset('../experiments/TE_cnig_provincias_no_embedding/risk_ds.nc')['risk_hat_ij']
    movement = risk_ij.copy()
    te_model = TEpydemic(movement, cases, embedding="round", n_bins=20)