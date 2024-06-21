#!/usr/bin/env python
# -*- coding: utf-8 -*-

############################################################################################################################
#
# download_fm_datasets.py: download Flow-Map datasets 
#
# Usage:
#  python download_fm_datasets.py --settings db_config.json --layer cnig_provincias --dataset covid_cases [--out outfolder]
#    i.e., the arguments <...> are optional and have defaults.
# 
# Dependencies include pandas and pymongo
#
# Examples (run from directory containing the .svg files):
#  python download_fm_datasets.py --settings db_config.json --layer cnig_provincias --dataset daily_mobility --out data/
#  python download_fm_datasets.py --settings db_config.json --layer abs_09 --dataset zone_movements --out data/
#  python download_fm_datasets.py --settings db_config.json --layer zon_bas_13 --dataset covid_cases --out data/
#
# Author: Miguel Ponce de León
#
############################################################################################################################

__author__ = "Miguel Ponce de León"
__email__  = "miguel.ponce@bsc.es"

import sys
import os
import json
import argparse
import pytz
from datetime import timezone

import pymongo
import pandas as pd

from itertools import product


def get_geolayer(layer):
    collection = "layers"

    if layer == "cnig_provincias":
        field_name = "rotulo"
    elif layer == "abs_09":
        field_name = "NOMABS"
    elif layer == "zon_bas_13":
        field_name = "DESBDT"
    
    pipeline = [ { "$match": { "layer": layer } } ]
    cursor = data_base[collection].aggregate(pipeline)
    geojson_dict = {"type": "FeatureCollection", "features": []}
    for doc in cursor:  
        entry = { 
            "id": doc["id"],
            "centroid": doc["centroid"],
            "type": doc["feat"]["type"],
            "geometry": doc["feat"]["geometry"],
            "name": doc["feat"]["properties"][field_name]
        }
        geojson_dict["features"].append(entry)

    return geojson_dict

def get_covid_cases(layer, ev, start_date=None, end_date=None):
    collection = "layers.data.consolidated"
    pipeline = [
        {
            "$match": {"layer": layer, "type": "covid19", "ev": ev}
        }, 
        {
            "$project": {
                "date"       : 1,
                "id"         : 1,
                "new_cases"  : 1,
                "total_cases": 1
            }
        }
    ]
    cursor = data_base[collection].aggregate(pipeline)
    df = pd.DataFrame([row for row in cursor])
    df = df.drop(["_id"], axis=1)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index(["date", "id"])
    df = df.sort_index()
    return df

def get_daily_mobility_matrices(layer, start_date=None, end_date=None):
    collection = "mitma_mov.daily_mobility_matrix"
    pipeline = [
        {
            "$match": {
                "source_layer": layer, 
                "target_layer": layer, 
            }
        }, 
        {
            "$project": {
                "date": 1,
                "source" : 1,
                "target" : 1,
                "trips": 1
            }
        }
    ]
    cursor = data_base[collection].aggregate(pipeline)
    df = pd.DataFrame([row for row in cursor])
    df = df.drop(["_id"], axis=1)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index(["date", "source", "target"])
    df = df.sort_index()

    return df  

def get_zone_movements(layer, start_date=None, end_date=None):
    collection = "layers.data.consolidated"
    pipeline = [
        {
            "$match": { "layer": layer, "type": "zone_movements"}
        }, 
        {
            "$project": {
                "date"    : 1,
                "id"      : 1,
                "viajes"  : 1,
                "personas": 1
            }
        }
    ]

    base_columns = ["date", "id", "personas", "viajes"]
    trips_col_rename  = {0:"0", 1:"1", 2:"2", -1:">2" }
    trips_columns = ["0", "1", "2", ">2"]

    cursor = data_base[collection].aggregate(pipeline)
    df = pd.DataFrame([row for row in cursor])
    df = df.drop(["_id"], axis=1)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df[base_columns].pivot(index=("date", "id"), columns="viajes", values="personas")
    df = df.rename(columns=trips_col_rename)
    df = df[trips_columns]

    df.loc[:, "total_population"] = df[trips_columns].sum(axis=1)
    df.loc[:, "moving_population"] = df[trips_columns[1:]].sum(axis=1)
    df = df[["total_population", "moving_population"]]
    df.columns.name = "population"
    df = df.sort_index()

    return df

def fill_missing_entries(df, dim):
    assert dim >= 2
    assert dim <= 3
    start_date  = df.index.get_level_values("date").min()
    end_date    = df.index.get_level_values("date").max()
    date_range  = pd.date_range(start=start_date, end=end_date)
    n_dates     = date_range.shape[0]
    
    if dim == 3:
        patches_ids = df.index.get_level_values("source").unique()
        n_patches   = patches_ids.shape[0]
        if n_dates * n_patches**2 == df.shape[0]:
            return df
        print(f"Filling not recorded combinations (date, origin, destination) using zeros")
        full_index  = list(product(date_range, patches_ids, patches_ids))
    elif dim == 2:
        patches_ids = df.index.get_level_values("id").unique()
        n_patches   = patches_ids.shape[0]
        if n_dates * n_patches == df.shape[0]:
            return df
        full_index  = list(product(date_range, patches_ids))
        print(f"Filling not recorded combinations (date, id) using zeros")
    
    df = df.reindex(full_index, fill_value=0)
    return df 

def create_parser():

    parser = argparse.ArgumentParser(description="Download daily mobility matrices for a given layer")

    parser.add_argument("--settings", dest="settings_json", action="store", required=True,
                        help="JSON file with MongoDB settings")
    parser.add_argument("--dataset", action="store", dest="dataset", required=True,
                        choices=DATASETS, help="The dataset to download")

    parser.add_argument("--layer", action="store", dest="layer", required=True,
                        choices=LAYERS, help="The Layer define the set of geographic patches")

    parser.add_argument("--out", action="store", dest="output_folder", default=".",
                         help="Output folder to store results")

    parser.add_argument("--fill_missing", action="store", dest="fill_missing", default=True,
                         help="Fill combinations (date, origin, destination) not present with zeros")

    parser.add_argument("--sdate", action="store", dest="start_date", default=None,
                         help="Starting date for daily mobility (defulat None)")

    parser.add_argument("--edate", action="store", dest="end_date", default=None,
                         help="Starting date for daily mobility (defulat None)")

    return parser


LAYERS    = ("cnig_provincias", "abs_09", "zon_bas_13")
DATASETS  = ("daily_mobility", "zone_movements", "covid_cases", "geolayer")


data_base = None
layer2covid_ev = {
    "cnig_provincias": "ES.covid_cpro",
    "abs_09": "09.covid_abs",
    "zon_bas_13": "13.covid_abs"
    }


def main():
    global data_base

    parser        = create_parser()
    args          = parser.parse_args()
    settings_json = args.settings_json
    layer         = args.layer
    dataset       = args.dataset
    fill_missing  = args.fill_missing
    output_folder = args.output_folder
    fname_out     = f"{layer}_{dataset}.parquet"
    fname_out     = os.path.join(output_folder, fname_out)

    with open(settings_json) as fh:
        client_parameters = json.load(fh)

    timezone      = "Europe/Madrid"
    tz            = pytz.timezone(timezone)
    database_name = "FlowMaps"
    client        = pymongo.MongoClient(**client_parameters)
    codec_options = client.codec_options.with_options(tzinfo=tz, tz_aware=True)
    data_base     = pymongo.database.Database(client, database_name, codec_options=codec_options)

    dim = -1
    if dataset == "geolayer":
        fname_out = f"{layer}.geojson"
        fname_out     = os.path.join(output_folder, fname_out)
        print(f"Downloading geolayer {layer}...", end=" ")
        geojson_dict = get_geolayer(layer)
        print("Done!")
        print(f"Writing results to {fname_out}")
        with open(fname_out, 'w') as fh:
            json.dump(geojson_dict, fh)
        sys.exit(0)

    elif dataset == "daily_mobility":
        print(f"Downloading daily mobility matrices layer={layer}...", end=" ")
        df = get_daily_mobility_matrices(layer)
        dim = 3
        patches_ids  = df.index.get_level_values("source").unique()
    elif dataset == "zone_movements":        
        print(f"Downloading zone movements dataset layer={layer}...", end=" ")
        df = get_zone_movements(layer)
        dim = 2
        patches_ids  = df.index.get_level_values("id").unique()
    elif dataset == "covid_cases":
        cases_ev = layer2covid_ev[layer]
        df = get_covid_cases(layer, cases_ev)
        print(f"Downloading COVID19 dataset layer={layer} ev={cases_ev}...", end=" ")
        dim = 2
        patches_ids  = df.index.get_level_values("id").unique()
    else:
        patches_ids = None
    
    print("Done!")

    start_date   = df.index.get_level_values("date").min()
    end_date     = df.index.get_level_values("date").max()
    date_range   = pd.date_range(start=start_date, end=end_date)
    
    n_dates      = date_range.shape[0]
    n_patches    = patches_ids.shape[0]

    print(f"Total patches {n_patches}")
    print(f"Total dates {n_dates}")
    print(f"Total entries {df.shape[0]}")

    if fill_missing:
        df = fill_missing_entries(df, dim)
        if dim == 3:
            assert n_dates * n_patches**2 == df.shape[0]
        elif dim == 2:
            assert n_dates * n_patches == df.shape[0]

    print(f"Writing dataframe to {fname_out}")
    df.to_parquet(fname_out)

if __name__ == "__main__":
    main()