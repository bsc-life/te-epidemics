#!/usr/bin/env python
# -*- coding: utf-8 -*-

############################################################################################################################
#
# calculater_transfer_entropy.py: calculate transfer entropy between mobility associated risk and cases
#
# Usage:
#  python calculater_transfer_entropy.py --config te_params.json --layer cnig_provincias [--out outfolder] [--cpus nun_of_cpus]
# 
# Dependencies include pandas and pymongo
#
# Examples (run from directory containing the .svg files):
#  python calculater_transfer_entropy.py base_folder --layer cnig_provincias --out experiments --cpus 8
#
# Author: Miguel Ponce de León
#
############################################################################################################################

__author__ = "Miguel Ponce de León"
__email__  = "miguel.ponce@bsc.es"

import os
import sys
import json
import math
import argparse

import numpy as np
import xarray as xr
import pandas as pd

from datetime import date
import multiprocessing as mp

from itertools import product, permutations
from pyinform import transfer_entropy


#########################################################################
# Functions for encoding data into a symbolic embedding
#########################################################################

def symbolic_embedding(x, m): 
    symbols_dict = {tuple(s):i for i,s in enumerate(list(permutations(range(m))))}
    L = x.shape[0]
    y = np.zeros(L-m+1)
    for i in range(x.shape[0]-m+1):
        values = tuple(x[i:i+m])
        if len(set(values)) == m:
            ixd = tuple(np.argsort(values))
            s = symbols_dict[ixd]
        else: # seven possible situations
            s = pathological_cases(values, m)
        y[i] = s
    return np.array(y, dtype=int)

def encode_data(data, embedding="round", nbins=20, scale=100, symmetric_binning=True, m=3):
    xa_encoded_data = None

    #####################
    # ROUND EMBEDDING
    #####################
    if embedding == "round":
        xa_encoded_data = data.round()
        
    #####################
    # UNIFORM EMBEDDING
    #####################
    if embedding == "uniform":
        bins = np.linspace(0, scale,  nbins)
        # ENCODE CASES DATA DIM=2
        if len(data.dims) == 2:
            if symmetric_binning:
                XMax = float(data.max().values)
            else:
                XMax = data.max('date').values
            data = (data.T / XMax).T * scale
        # ENCODE RISK/MOVILITY DATA DIM=3
        elif len(data.dims) == 3:
            XMax = data.max('date').values
            if symmetric_binning:
                XMax[XMax-XMax.T < 0] = 0
                XMax += XMax.T
                np.fill_diagonal(XMax, 1)
            data = data.transpose('date', 'source', 'target') / XMax * scale
            data = data.transpose('source', 'target', 'date')
        
        my_digitize = lambda x,bins: np.digitize(x, bins, right=False)
        xa_encoded_data = xr.apply_ufunc(my_digitize, data, bins)

    #####################
    # SYMBOLIC EMBEDDING
    #####################
    if embedding == "symbolic":
        # ENCODE CASES DATA DIM=2
        if len(data.dims) == 2:
            patches_ids = data.coords['id'].values
            n_patches = patches_ids.shape[0]
            date_range = data.coords['date'].values
            n_dates = date_range.shape[0]
            encoded_data = np.ndarray((n_patches,n_dates-m+1))
            for i,s in enumerate(patches_ids):
                x = data.loc[s,:].values
                encoded_data[i,:] = symbolic_embedding(x, m)
            encoded_data = encoded_data.astype(int)
            xa_encoded_data = xr.DataArray(encoded_data, coords={'id': patches_ids, 'date':date_range[m-1:]})
        # ENCODE RISK/MOVILITY DATA DIM=3
        elif len(data.dims) == 3:
            patches_ids = data.coords['source'].values
            n_patches = patches_ids.shape[0]
            date_range = data.coords['date'].values
            n_dates = date_range.shape[0]
            encoded_data = np.ndarray((n_patches,n_patches,n_dates-m+1))
            for i,s in enumerate(patches_ids):
                for j,t in enumerate(patches_ids):
                    x = data.loc[s,t,:].values
                    encoded_data[i,j,:] = symbolic_embedding(x, m)

            encoded_data = encoded_data.astype(int)
            xa_encoded_data = xr.DataArray(encoded_data, coords={'source': patches_ids, 'target':patches_ids, 'date':date_range[m-1:]})
            
    return xa_encoded_data

#########################################################################
# Functions for calculating the transfer of entropy
#########################################################################

def error_callback(e):
    raise Exception("Error accoured while calculating TE")

def log_result(result):
    RESULT_LIST.append(result)

def calculate_te(source, target, TE_x="risk_ij", s=1, omega=21, delta=5, k=1, **kargs):
    if TE_x=="risk_ij":
        X_ij = X.loc[source, target, :].to_pandas()
    elif TE_x=="risk_hat_ij":
        X_ij = X.loc[source, target, :].to_pandas()
    elif TE_x=="new_cases_by_100k":
        X_ij = Y.loc[source, :].to_pandas()
    elif TE_x=="new_cases":
        X_ij = Y.loc[source, :].to_pandas()
    else:
        raise Exception(f"Unknown TEx param {TEx}")

    Y_j = Y.loc[target, :].to_pandas()
        
    assert X_ij.shape[0] == Y_j.shape[0]
    
    L = Y_j.shape[0]
    N = L - (omega + delta) + 1
    n_points = int( N / s )
    TE = np.zeros(n_points)
        
    # Filter self loops
    if source == target: 
        return (source, target, TE)
    # Speed-up to avoid calculating cases where there is no risk between i and j
    if X_ij.max() == 0:
        return (source, target, TE)
    
    for j,i in enumerate(range(0, n_points, s)):
        x = X_ij[i:i+omega]
        y = Y_j[i+delta:i+delta+omega]
        TE[j] = transfer_entropy(x, y, k)
    
    return (source, target, TE)

def results2xarray(data, patches_ids, date_range):
    i, j, te_ij = data[0]
    m = te_ij.shape[0]
    n = len(patches_ids)
    nda = np.ndarray(shape=(n,n,m))
    patch_indexer = {k:i for i,k in enumerate(patches_ids)}
    for source,target,entropy in data:
        i = patch_indexer[source]
        j = patch_indexer[target]
        if np.isinf(entropy).any():
            entropy[np.isinf(entropy)] = 0
        if np.isnan(entropy).any():
            entropy[np.isnan(entropy)] = 0
        nda[i, j, :] = entropy
    TE_da = xr.DataArray(nda, dims=("source", "target", "date"), 
                         coords={"source": patches_ids, "target": patches_ids, "date": date_range})
    return TE_da

def create_parser():

    parser = argparse.ArgumentParser(description="Calculate transfer entropy between MAR and cases.")
    
    parser.add_argument("base_folder", help="Folder where the processed data is located")

    parser.add_argument("-w", "--omega", action="store", dest="omega", default=21, type=int,
                         help="Omega parameter for the TE. is the lenght of the sliding window")
    
    parser.add_argument("-d", "--delta", action="store", dest="delta", default=7, type=int,
                         help="Detal (delay) parameter for the TE")
    
    parser.add_argument("-s", dest="s", default=1, type=int,
                         help="s parameter for the TE. Is the offset of the sliding window")

    parser.add_argument("-k", default=1, type=int,
                         help="k parameter for the TE. Is the memory of the process")
    
    parser.add_argument("-e", "--embedding", dest="embedding", action="store", default="round",
                         help="Type of embedding used to convert data", choices=EMBEDDINGS)
    
    parser.add_argument("-b", "--nbins", action="store", dest="nbins", default=20, type=int,
                         help="Nº of bins to use in the uniform embeding")

    parser.add_argument("--symmetric_binning", action="store_true",
                         help="Wheather to perform or not a symmetric (uniform) binning")
   
    parser.add_argument("-x", "--TEx", dest="TE_x", action="store", default="risk_ij",
                        help="Variable measured at the source (X)", choices=TE_X_CHOICES)
    
    parser.add_argument("-y", "--TEy", dest="TE_y", action="store", default="new_cases_by_100k",
                        help="Variable measured at the destination (Y)", choices=TE_Y_CHOICES)
    
    parser.add_argument("-c", "--cpus", action="store", dest="cpus", default=None, type=int,
                         help="Nº of cpus use for parallel calculation")

    parser.add_argument("--debug", action="store_true",
                         help="Run a single TE calculation for debuggin prouposes")

    return parser


##################################################

EMBEDDINGS   = ("uniform", "round", "symbolic")
TE_X_CHOICES = ("risk_ij", "risk_hat_ij", "new_cases_by_100k", "new_cases")
TE_Y_CHOICES = ("new_cases_by_100k", "new_cases")

##################################################

X = None
Y = None
RESULT_LIST = None

def main():
    global X
    global Y
    global RESULT_LIST

    cur_dir       = os.path.dirname(__file__)
    project_root  = os.path.join(cur_dir, "../")
    project_root  = os.path.realpath(project_root)

    parser      = create_parser()
    args        = parser.parse_args()
    base_folder = args.base_folder
    cpus        = args.cpus
    debug       = args.debug

    TE_x        = args.TE_x
    TE_y        = args.TE_y

    embedding_params = {}
    embedding_params["embedding"]         = args.embedding
    embedding_params["nbins"]             = args.nbins
    embedding_params["symmetric_binning"] = args.symmetric_binning

    te_params = {}
    te_params["k"]         = args.k
    te_params["omega"]     = args.omega
    te_params["delta"]     = args.delta
    te_params["s"]         = args.s
    te_params["TE_x"]      = args.TE_x
    te_params["TE_y"]      = args.TE_y
    
    # Getting the data folder
    base_folder = os.path.join(project_root, base_folder)
    assert os.path.exists(base_folder)
    
    cases_ds_fname = os.path.join(base_folder, "cases_pop_ds.nc")
    assert os.path.exists(cases_ds_fname)
    print(f"- Reading Cases data records from {cases_ds_fname}")
    cases_ds = xr.load_dataset(cases_ds_fname)
    
    risk_ds_fname  = os.path.join(base_folder, "risk_ds.nc")
    assert os.path.exists(risk_ds_fname)
    print(f"- Reading Risck data records from {risk_ds_fname}")
    risk_ds  = xr.load_dataset(risk_ds_fname)

    # assert TE_x in risk_ds
    assert TE_y in cases_ds
    assert cases_ds.coords['date'].shape == risk_ds.coords['date'].shape
    assert cases_ds.coords['id'].shape   == risk_ds.coords['source'].shape
    assert cases_ds.coords['id'].shape   == risk_ds.coords['target'].shape

    params_strn = f"w{args.omega}_d{args.delta}_{str.upper(args.embedding[0])}"
    if args.embedding == "uniform":
        aux = str.upper(str(args.symmetric_binning)[0])
        params_strn += f"_nbin{args.nbins}_sb{aux}"

    params_strn += f"_{args.TE_x}"
    print(f"- Creating output folder for storing run outpus, config, etc:")
    today = date.today()
    strn_date = today.strftime("%Y%m%d")
    output_folder = os.path.join(base_folder, f"run_{params_strn}")
    count = 1
    while os.path.exists(output_folder):
        count += 1
        output_folder = os.path.join(base_folder, f"run_{params_strn}_{count}")

    output_folder = os.path.abspath(output_folder)
    print(f"\t{output_folder}")
    
    os.mkdir(output_folder)
    params_fname = os.path.join(output_folder, "params.json")
    te_ds_fname  = os.path.join(output_folder, "TE.nc")

    ###################################################################
    # Calculating TE
    ###################################################################
    if cpus is None:
        cpus = mp.cpu_count()
    
    patches_ids = cases_ds.coords["id"].values
    date_range  = cases_ds.coords["date"].to_pandas().index
    start_date  = date_range[0]
    end_date    = date_range[-1]

    print(f"Encoding data using \"{args.embedding}\" embedding")
    if TE_x == "risk_hat_ij":
        X = encode_data(risk_ds[TE_x] * 5, **embedding_params)
    elif TE_x == "risk_ij":
        X = encode_data(risk_ds[TE_x], **embedding_params)
    else:
        X = encode_data(cases_ds[TE_y], **embedding_params)

    Y = encode_data(cases_ds[TE_y], **embedding_params)

    print(f"- Calculating Transfer Entropy between {len(patches_ids)} zones from {start_date} to {end_date}")
    print(f"- TE_xy using {TE_x} as variable in X and {TE_y} as variable in Y")
    print(f"- Parameters used:")
    for k,v in te_params.items():
        print(f"\t* {k}: {v}")

    RESULT_LIST = []
    if debug:
        print("- Running debugg")
        i = "28"; j = "33"
        print(f"- calculating TE({i}->{j}):")
        i, j, te_ij = calculate_te(i, j, **te_params)
        # i, j, te_ij = calc_te(i, j, date_range, **te_params)
        print(f"\t{i} -> {j} = {te_ij.sum()}\n")
        sys.exit(0)
    
    print(f"- Computing TE... ", end=" ")
    if cpus > 1:
        print(f"Running in parallel using {cpus} cpus.")
        pool = mp.Pool(cpus)
        for i,j in product(patches_ids, repeat=2):
            pool.apply_async(calculate_te, args=(i, j), kwds=te_params, 
                            callback=log_result, error_callback=error_callback)
        pool.close()
        pool.join()
    else:
        print("Running sequentialy.")
        for i, j in product(patches_ids, repeat=2):
            i, j, te_ij = calculate_te(i, j, **te_params)
            print(f"- calculating TE({i}->{j}):")
            print(f"\t{i} -> {j} = {te_ij.sum()}\n")
            RESULT_LIST.append((i, j, te_ij))

    print("\t* Done!")

    # correct offset of the TE by shifting dates to (w-d)
    i,j,te_ij = RESULT_LIST[0]
    n = te_ij.shape[0]
    omega = te_params["omega"]
    delta = te_params["delta"]
    start = int(omega)
    TE_dates = date_range[start:]
    end = min(len(TE_dates), te_ij.shape[0])
    TE_dates = TE_dates[:end]

    TE_ds = results2xarray(RESULT_LIST, patches_ids, TE_dates)
    total_te = TE_ds.sum(dim="date")
    print(f"\t* Total Entropy Transfered: {total_te.values.sum():.2f}")

    print(f"\t* Writing final results to {te_ds_fname}")
    TE_ds.to_netcdf(te_ds_fname)
    print(f"\t* Writing config to {params_fname}")

    params_dict = {"te_params": te_params, "embedding": embedding_params}
    with open(params_fname, "w") as fh:
        json.dump(params_dict, fh)

if __name__ == "__main__":
    main()
