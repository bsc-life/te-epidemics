#!/usr/bin/env python
# -*- coding: utf-8 -*-

###################################################################################
#
# calculate_SIR_model.py: simulate epidemic process with Epicommute SIR model and
# save cases and risk datasets
#
# Authors: Camila F. T. Pontes, Miguel Ponce de Leon
#
###################################################################################

__author__ = "Camila F. T. Pontes"
__email__  = "camila.pontes@bsc.es"

import os
import sys
import uuid
import json
import jsonschema
import argparse

import numpy as np
import xarray as xr
import pandas as pd
from itertools import combinations, product, permutations

import multiprocessing as mp

from EpiCommute import SIRModel

def validate_config_parameters(config_dict):
    configSchema = {
        "type": "object",
        "properties": {
            "output_folder": {"type": "string"},
            "mobility_matrix_file": {"type": "string"},
            "n_replicates": {"type": "number"},
            "model_params": {
                "type": "object",
                "properties":{
                    "t_max": {"type": "number"},
                    "i0": {"type": "number"},
                    "dt": {"type": "number"},
                    "dt_save": {"type": "number"},
                    "r0": {"type": "number"},
                    "mu": {"type": "number"},
                    "hub_idx": {"type": "number"},
                },
            },
        },
        "required": ["output_folder", "mobility_matrix_file"]
    }

    jsonschema.validate(config_dict, configSchema)

    assert config_dict["model_params"]["t_max"] > 0
    assert config_dict["model_params"]["i0"] >= 1
    assert config_dict["model_params"]["dt"] > 0
    assert config_dict["model_params"]["dt_save"] >= config_dict["model_params"]["dt"]
    assert config_dict["model_params"]["r0"] > 0
    assert config_dict["model_params"]["mu"] > 0
    assert config_dict["model_params"]["hub_idx"] >= 0
    assert config_dict["n_replicates"] >= 1

    return True

def create_parser():

    parser = argparse.ArgumentParser(description="Simulate epidemic using the EpiCommute SIR model.")

    parser.add_argument("output_folder", help="Folder where the processed data is located")

    parser.add_argument("mobility_matrix_file", help="Mobility matrix file")

    parser.add_argument("-dt", dest="dt", default=0.1, type=float,
                         help="Simulation timestep")

    parser.add_argument("-tmax", dest="t_max", default=300, type=int,
                        help="Simulation time")

    parser.add_argument("-dt_save", dest="dt_save", default=1, type=float,
                         help="Time interval to save the simulation")

    parser.add_argument("-i0", dest="i0", default=10, type=int,
                         help="s parameter for the TE. Is the offset of the sliding window")

    parser.add_argument("-r0", dest="r0", default=1.5, type=float,
                         help="Infection rate")

    parser.add_argument("-mu", dest="mu", default=1, type=float,
                         help="mu")

    parser.add_argument("-hub", dest="hub_idx", default=0, type=int,
                        help="Index of hub node where the epidemic starts")

    parser.add_argument("-n", "--nreplicates", dest="n_replicates", default=500, type=int,
                         help="Number of simulation replicates")

    parser.add_argument("-c", "--cpus", action="store", dest="cpus", default=8, type=int,
                         help="Nº of cpus use for parallel calculation")

    return parser

def run_SIR_model(i, model):
    model.reset_initialize_simulation()
    result = model.run_simulation()

    return result

def error_callback(e):
    print("Error accoured while calculating TE", e)

def log_result(result):
    cases_density = np.array(result['I'])
    arrival_time = np.array(result['T_arrival'])
    time = np.array(result['t'])
    RESULT_LIST.append((cases_density, arrival_time, time))

RESULT_LIST = None

def main():
    global RESULT_LIST

    # this is used to rescale small values of the Risk_ij_hat
    density_scale = 1e2 

    cur_dir        = os.path.dirname(__file__)
    project_root   = os.path.join(cur_dir, "../")
    project_root   = os.path.realpath(project_root)

    parser         = create_parser()
    args           = parser.parse_args()

    base_outfolder = args.output_folder
    mobility_matrix_file = args.mobility_matrix_file
    cpus        = args.cpus
    n_replicates = args.n_replicates

    model_params = {}
    model_params["dt"]          = args.dt
    model_params["t_max"]       = args.t_max
    model_params["dt_save"]     = args.dt_save
    model_params["i0"]          = args.i0
    model_params["mu"]          = args.mu
    model_params["r0"]          = args.r0
    model_params["hub_idx"]     = args.hub_idx

    topology = args.mobility_matrix_file.split("_")[-2]
    output_folder = os.path.join(base_outfolder, f"SIR_simulation_{topology}")
    output_folder = os.path.abspath(output_folder)

    print(f"- Beginning data processing for SIR model using {args}")
    print(f"- Creating output folder for storing run outpus, config, etc:")
    if os.path.exists(output_folder):
        print(f"- WARNING output folder {output_folder} already exists, results will be overwriten")
    else:
        os.mkdir(output_folder)
        print(f"\t{output_folder}")

    config_fname    = os.path.join(output_folder, "config.json")
    cases_ds_fname  = os.path.join(output_folder, "cases_pop_ds.nc")
    risk_ds_fname   = os.path.join(output_folder, "risk_ds.nc")

    cpus           = args.cpus
    if cpus is None:
        cpus = mp.cpu_count()

    config = {}
    config["output_folder"] = output_folder
    config["mobility_matrix_file"] = mobility_matrix_file
    config["n_replicates"] = n_replicates
    config["model_params"] = model_params
    validate_config_parameters(config)

    mobility = np.load(mobility_matrix_file)
    subpopulation_sizes = np.sum(mobility, axis=0)
    N = len(subpopulation_sizes)
    
    # Initialize the model
    model = SIRModel(
                mobility,
                subpopulation_sizes,
                outbreak_source = args.hub_idx,     # random outbreak location
                dt = args.dt,                       # simulation time interval
                dt_save = args.dt_save,             # time interval when to save observables
                I0 = args.i0,                       # number of initial infected
                VERBOSE = False,                    # print verbose output
                mu =  args.mu,
                R0 = args.r0,
                T_max = args.t_max,
                save_observables = ['epi_subpopulations', 'arrival_times']
            )
    model.population

    replicates = []
    arrival = []

    RESULT_LIST = []
    print(f"- Running SIR model simulations... ", end=" ")
    if cpus > 1:
        print(f"Running in parallel using {cpus} cpus.")
        pool = mp.Pool(cpus)
        for i in range(n_replicates):
            pool.apply_async(run_SIR_model, args=(i,), kwds={"model":model},
                            callback=log_result, error_callback=error_callback)
        pool.close()
        pool.join()
    else:
        print("Running sequentialy.")
        for i in range(n_replicates):
            result = run_SIR_model(i, model)
            log_result(result)

    print("\t* Done!")

    #time = RESULT_LIST[0][2]
    #time = np.round(time).astype(int)
    dates = pd.date_range(start='2020-03-01', periods=args.t_max+1)
    patches = ['H'] + [f"L{i}" for i in range(1,N)]

    data = np.zeros((len(dates), N, len(RESULT_LIST)),dtype=float)

    LAG = 7
    for i,rep in enumerate(RESULT_LIST):
        for j in range(N):
            data[LAG:,j,i] = rep[0][:-LAG,j] #* subpopulation_sizes[j]
        
    da_inf = xr.DataArray(data, coords=[dates, patches, range(1,n_replicates+1)], dims=["date", "id", "rep"])

    da_subpop  = xr.DataArray(subpopulation_sizes, coords=[patches], dims=["id"])

    da_cases = da_inf.mean(dim="rep") 
    da_cases = da_cases.transpose("id", "date")
   
    mobility_exp = mobility+mobility.T
    np.fill_diagonal(mobility_exp, 0)
    print(mobility_exp)

    mobility_exp = np.repeat(mobility_exp[np.newaxis, :, :], len(dates), axis=0)
    mobility_exp = mobility_exp.transpose(1, 2, 0)
    da_trip = xr.DataArray(mobility_exp, coords=[patches, patches, dates], dims=["source", "target", "date",])    
    
    da_risk = da_trip * da_cases.rename({"id":"source"})
    da_risk_hat = da_risk / da_subpop.rename({'id': 'target'})
    da_risk_hat *= density_scale


    cases_ds = xr.Dataset(data_vars={"new_cases":da_cases})
    cases_ds["new_cases_by_100k"] = da_cases / da_subpop * 100000
    print(cases_ds)

    risk_ds = xr.Dataset(data_vars={"risk_ij":da_risk, "risk_hat_ij":da_risk_hat})
    print(risk_ds)

    print(f"\t* Writing Cases in NetCDF to {cases_ds_fname}")
    cases_ds.to_netcdf(cases_ds_fname)

    print(f"\t* Writing Risk in NetCDF to {risk_ds_fname}")
    risk_ds.to_netcdf(risk_ds_fname)

    ###################################################################
    # Creating the three dataframes required for calculating TE
    ###################################################################

    print(f"\t* Writing config to {config_fname}")
    with open(config_fname, "w") as fh:
        json.dump(config, fh, indent=True)

if __name__ == "__main__":
    main()
