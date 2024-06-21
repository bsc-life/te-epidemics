#!/usr/bin/env python
# -*- coding: utf-8 -*-

############################################################################################################################
#
# calculater_transfer_entropy.py: calculate transfer entropy between mobility associated risk and cases
#
# Usage:
#  python preprocess_datasets.py --config config.json --layer cnig_provincias [--out outfolder]
# 
# Dependencies include pandas and pymongo
#
# Examples (run from directory containing the .svg files):
#
#
# Author: Miguel Ponce de León
#
############################################################################################################################

__author__ = "Miguel Ponce de León"
__email__  = "miguel.ponce@bsc.es"

import os
import sys
import uuid
import json
import jsonschema
import argparse

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

import seaborn as sns
import matplotlib.dates as mdates

import multiprocessing as mp

from itertools import product, permutations

from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
from pyinform import transfer_entropy

cur_dir = os.path.dirname(__file__)
custom_modules_path = os.path.join(cur_dir, "../python")
sys.path.append(custom_modules_path)



def validate_config_parameters(config_dict):
    configSchema = {
        "type": "object",
        "properties": {
            "data_folder": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "filter_patches": {"type": "array"},
            "cases_rolling_window": {"type": "number"},
            "trips_rolling_window": {"type": "number"},
            "active_cases_days": {"type": "number"},
            "risk_normalization_var": {"type": "string"},
            "TE_x_variable": {"type": "string"},
            "TE_y_variable": {"type": "string"},
            "te_params": {"type": "object"}
        },
        "required": [
            "data_folder", "start_date", "end_date",
            "active_cases_days", "risk_normalization_var"
            ]
    }
    
    jsonschema.validate(config_dict, configSchema)
    
    assert config_dict["cases_rolling_window"] >= 0
    assert config_dict["trips_rolling_window"] >= 0
    assert config_dict["active_cases_days"] >= 0
    assert config_dict["risk_normalization_var"] in ("total_population", "moving_population")
    
    return True



def create_parser():

    parser = argparse.ArgumentParser(description="Calculate transfer entropy between MAR and cases.")

    parser.add_argument("--config", dest="config_json", action="store", required=True,
                        help="JSON file with config")

    parser.add_argument("--layer", action="store", dest="layer", required=True,
                    choices=LAYERS, help="The Layer define the set of geographic patches")
    
    parser.add_argument("--out", action="store", dest="output_folder", default=".",
                         help="Output folder to store results")

    return parser


LAYERS      = ("cnig_provincias", "abs_09", "zon_bas_13")

def main():
    cur_dir        = os.path.dirname(__file__)
    project_root   = os.path.join(cur_dir, "../")
    project_root   = os.path.realpath(project_root)
    
    risk_hat_scale = 1e5

    parser         = create_parser()
    args           = parser.parse_args()
    base_outfolder = args.output_folder
    layer          = args.layer

    config = None
    with open(args.config_json) as fh:
        config = json.load(fh)

    validate_config_parameters(config)

    # Getting the data folder
    data_folder = os.path.join(project_root, config["data_folder"])
    assert os.path.exists(data_folder)

    output_folder = os.path.join(base_outfolder, f"TE_{layer}")
    output_folder = os.path.abspath(output_folder)
    print(f"- Beginning data processing for layer {layer} using {args.config_json}")
    print(f"- Creating output folder for storing run outpus, config, etc:")
    
    base_output_folder = output_folder
    i = 0
    while os.path.exists(output_folder):
        i += 1
        print(f"- WARNING output folder {output_folder} already exists, trying with: " + f"{base_output_folder}_{i}")
        output_folder = f"{base_output_folder}_{i}"
    
    os.mkdir(output_folder)
    print(f"\t{output_folder}")
    

    config_fname    = os.path.join(output_folder, "config.json")
    cases_ds_fname  = os.path.join(output_folder, "cases_pop_ds.nc")
    risk_ds_fname   = os.path.join(output_folder, "risk_ds.nc")

    config["layer"]          = layer
    config["output_folder"]  = output_folder
    config["cases_ds_fname"] = cases_ds_fname
    config["risk_ds_fname"]  = risk_ds_fname

    ###################################################################
    ###################################################################
    ##            LOADING CONFIGURATION AND PARAMETERS
    ###################################################################    
    ###################################################################

    # Parameters
    active_cases_days      = config["active_cases_days"] # 7
    risk_normalization_var = config["risk_normalization_var"] # "total_population"
    start_date             = config["start_date"]
    end_date               = config["end_date"]

    # sizes of rolling windows used to smooth time series
    cases_rolling_window = config["cases_rolling_window"]
    mov_rolling_window   = config["trips_rolling_window"]

    # Creating path to all the datasets required for calculating the TE
    geojson_fname  = os.path.join(data_folder, f"{layer}.geojson")
    cases_fname    = os.path.join(data_folder, f"{layer}_covid_cases.parquet")
    trips_fname    = os.path.join(data_folder, f"{layer}_daily_mobility.parquet")
    movement_fname = os.path.join(data_folder, f"{layer}_zone_movements.parquet")
    
    ###################################################################
    ###################################################################
    ##                   LOADING REQUIRED DATASETS
    ###################################################################
    # #################################################################   
    
    print(f"\t* Loading patches from {os.path.relpath(geojson_fname)}")
    gdf_patches = gpd.read_file(geojson_fname)
    gdf_patches.set_index("id", inplace=True)
    gdf_patches.sort_index(inplace=True)
    patches_ids = set(gdf_patches.index)
    print(f"\t* Total Patches: {len(gdf_patches.index)}")

    print(f"\t* Loading cases from: {os.path.relpath(cases_fname)}")
    df_cases = pd.read_parquet(cases_fname)
    cases_patches = set(df_cases.index.get_level_values("id"))

    print(f"\t* Loading OD matrices from: {os.path.relpath(trips_fname)}")
    df_trips = pd.read_parquet(trips_fname)
    trips_patches = set(df_trips.index.get_level_values("source"))

    print(f"\t* Loading zone movements from: {os.path.relpath(movement_fname)}")
    df_movements = pd.read_parquet(movement_fname)
    
    ###################################################################
    #  Correcting PATCHES
    ###################################################################

    print(f"\t\t> Total Cases patches: {len(cases_patches)}")
    if len( cases_patches - patches_ids ) > 0:
        unknown_patches = cases_patches - patches_ids 
        print("WARNING: cases dataset contains unknown patches:", end=" ")
        print(" ".join(unknown_patches))
        print("REMOVING unknown patches from cases dataset")
        df_cases = df_cases[df_cases.index.isin(patches_ids, level=1)].sort_index()
        print(f"\t\t> Total Cases patches after update: {len(cases_patches)}")

    if len( patches_ids - cases_patches ) > 0:
        unknown_patches = patches_ids- cases_patches 
        print("WARNING: geodataset contains unknown patches:", end=" ")
        print(" ".join(unknown_patches))
        print("REMOVING unknown patches from patches geodataset")
        patches_ids = patches_ids - unknown_patches
        gdf_patches = gdf_patches.loc[sorted(patches_ids)]
        print(f"\t\t> Total patches after update: {len(cases_patches)}")

    print(f"\t\t> Total mobility patches {len(trips_patches)}")
    if len( set(trips_patches) - set(gdf_patches.index) ) > 0:
        unknown_patches = set(trips_patches) - set(gdf_patches.index)
        print("WARNING: trips dataset contains unknown patches:")
        print(" ".join(unknown_patches))
        print("REMOVING unknown patches from mobility dataset")
        df_trips = df_trips[df_trips.index.isin(patches_ids, level=1)]
        df_trips = df_trips[df_trips.index.isin(patches_ids, level=2)].sort_index()
        df_movements = df_movements[df_movements.index.isin(patches_ids, level=1)].sort_index()
        print(f"\t\t> Total mobility patches after update: {len(cases_patches)}")

    # FILTER PATCHES
    filter_patches = []
    for i in config["filter_patches"]:
        if i not in patches_ids:
            continue
        patches_ids.remove(i)
        filter_patches.append(i)

    if len(filter_patches) > 0:
        patches_strn = ", ".join(filter_patches)
        print(f"- Filtering patches: [{patches_strn}]")
        mask = ~df_cases.index.get_level_values("id").isin(filter_patches)
        df_cases = df_cases[mask]

        mask = ~df_movements.index.get_level_values("id").isin(filter_patches)
        df_movements = df_movements[mask]

        mask  = ~df_trips.index.get_level_values("source").isin(filter_patches)
        mask &= ~df_trips.index.get_level_values("target").isin(filter_patches)
        df_trips = df_trips[mask]
        
    ###################################################################
    # Creating indexers for dates and patches
    ###################################################################
    date_range    = pd.date_range(start=start_date, end=end_date)
    patches_ids   = sorted( set(gdf_patches.index) & set(df_movements.index.get_level_values("id")) )
    
    #####################################################################
    # Applying rolling means to smooth data 
    #####################################################################
 

    #####################################################################
    # Aligning dates on all datasets and checking dimesions consistency
    #####################################################################
    print("- Aligning dates on all datasets and check dimesions consistency")
    df_cases     = df_cases.loc[date_range]
    df_trips     = df_trips.loc[date_range]
    df_movements = df_movements.loc[date_range]

    n_dates      = len(date_range)
    n_patches    = len(patches_ids)

    assert n_dates * n_patches == df_cases.shape[0]
    assert n_dates * (n_patches**2) == df_trips.shape[0]
    assert n_dates * n_patches == df_movements.shape[0]

    print(f"\t* Total pactches {n_patches}")
    print(f"\t* Total dates {n_dates}")

    print(f"\t* Total cases entries: {df_cases.shape[0]} ({n_dates}x{n_patches})")
    print(f"\t* Total daily mobility entries: {df_trips.shape[0]} ({n_dates}x{n_patches}x{n_patches})")
    print(f"\t* Total zone movements entries: {df_movements.shape[0]} ({n_dates}x{n_patches})")


    ###################################################################
    ###################################################################
    # CREATING THE DS FOR THE RISK 
    ###################################################################
    ###################################################################
    df_movements = df_movements.reorder_levels(["date", "id"]).sort_index()
    df_cases     = df_cases.reorder_levels(["date", "id"]).sort_index()

    if (df_cases.min() < 0).any():
        print("WARNING NEGATIVE CASES FOUND!")
        print("Applying absolute value")
        df_cases = df_cases.abs()
    
    print("- Processing Cases time series")
    print("\t* Creating xarray dataset for cases and populations")
    df_cases = df_cases.reorder_levels(['id', 'date']).sort_index()
    df_movements = df_movements.reorder_levels(['id', 'date']).sort_index()

    vars_dict = {}
    for i in df_cases.columns:
        vars_dict[i] = (("id", "date"), df_cases[i].unstack())
    for i in df_movements.columns:
        vars_dict[i] = (("id", "date"), df_movements[i].unstack())
    cases_ds = xr.Dataset(data_vars=vars_dict, coords={"id": patches_ids, "date": date_range})
        
    if cases_rolling_window > 0:
        print(f"\t* Applying rolling mean on cases and populations data using a windows of length: {cases_rolling_window}")
        for i in vars_dict.keys():
            cases_ds[i] = cases_ds[i].rolling(date=cases_rolling_window, min_periods=1).mean()
    
    print(f"\t* Calculating cases by 100k inhabitants")
    cases_ds['new_cases_by_100k'] = 1e5 * cases_ds["new_cases"] / cases_ds['total_population'].mean(dim='date')
    print(f"\t* Calculating active cases using a rolling sum: window={active_cases_days}")
    cases_ds['active_cases'] = cases_ds['new_cases'].rolling(date=active_cases_days, min_periods=1).sum()
    print(f"\t* Calculating source density cases: cases_density_i = active_cases_i / total_population_i")
    cases_ds['cases_density'] = cases_ds['active_cases'] / cases_ds['total_population'].mean(dim='date')

    ###################################################################

    print("- Processing trips dataset to calculate mobility associated risk")
    print("\t* Creating xarray dataset for trips and risk")
    df_trips = df_trips.reorder_levels(['source', 'target','date']).sort_index()
    mask = df_trips.index.get_level_values('source') == df_trips.index.get_level_values('target')
    df_trips[mask] = 0
    data = df_trips.values.reshape(n_patches, n_patches, n_dates)
    data = data.transpose(1,0,2)
    coords = {"source": patches_ids, "target": patches_ids, "date": date_range}
    risk_ds = xr.Dataset(data_vars={'trips_ij': (("source", "target", "date"), data)}, coords=coords)

    if mov_rolling_window > 0:
        print(f"- Applying rolling mean on trips using a windows of length: {mov_rolling_window}")
        risk_ds['trips_ij'] = risk_ds['trips_ij'].rolling(date=mov_rolling_window, min_periods=1).mean()
    
    print("\t* Calculating risk between i --> j as follows:")
    print(f"\t  risk_ij = p_cases_i * trips_ij")
    risk_ds['risk_ij'] = risk_ds['trips_ij'] * cases_ds['cases_density'].rename({'id': 'source'})

    print(f"\t  risk_hat_ij = risk_ij / {risk_normalization_var}_j * {risk_hat_scale:.0e}")
    risk_ds['risk_ij_hat_j'] = risk_ds['risk_ij'] / cases_ds[risk_normalization_var].rename({'id': 'target'}).mean(dim='date')
    risk_ds['risk_ij_hat_j'] *= risk_hat_scale

    print(f"\t  risk_hat_ij = risk_ij / {risk_normalization_var}_i * {risk_hat_scale:.0e}")
    risk_ds['risk_ij_hat_i'] = risk_ds['risk_ij'] / cases_ds[risk_normalization_var].rename({'id': 'source'}).mean(dim='date')
    risk_ds['risk_ij_hat_i'] *= risk_hat_scale
    
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
