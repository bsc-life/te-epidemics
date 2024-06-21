#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Miguel Ponce de León"
__email__  = "miguel.ponce@bsc.es"


import os
import sys
import json
import subprocess
import argparse
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

def main():
    
    cmd = f"python"
    path_scripts = "scripts/calculate_transfer_entropy.py"

    config_json = "etc/te_config.json"
    layer = "cnig_provincias"

    output_folder = "experiments/sweep"

    param_grid = {
        "cases_rolling_window": [0, 7],
        "trips_rolling_window": [0, 7],
        "use_bg": [True, False],
        "omega": [7, 14, 21, 28],
        "delta": [1, 3, 7, 10, 14],
        "n_bins": [10, 15, 20, 30, 50]
    }
    
    n_sets = 1
    for i in param_grid.values():
        n_sets *= len(i)

    print(f"- Running TE parameter sweep of {n_sets} combinations on layer {layer}")
    print(f"- Reading config template from {config_json}")
    with open(config_json) as fh:
        config = json.load(fh)

    print()
    print(f"- Creating parameter grid using (total {n_sets}):")
    for k,v in param_grid.items():
        print(f"\t{k}:", v)
    print

    all_param_sets = list(ParameterGrid(param_grid))
    print("- Starting the calculation")
    for param_set in tqdm(all_param_sets):
        new_config = config.copy()
        for k in new_config.keys():
            if k not in param_set:
                continue
            new_config[k] = param_set[k]

        for k in new_config['te_params'].keys():
            if k not in param_set:
                continue
            new_config['te_params'][k] = param_set[k]
    
        config_fname = "/tmp/config.json"
        with open(config_fname, "w") as fh:
            json.dump(new_config, fh)

        result = subprocess.run([cmd, path_scripts, "--config", config_fname, "--layer", layer,  "--out", output_folder], capture_output=True, text=True)
        
        if len(result.stderr):
            print(result.stderr)
            sys.exit(1)

        line = result.stdout.rstrip().split("\n")[-1]
        instance_folder = line.split("/")[-2]
        instance_folder = os.path.join(output_folder, instance_folder)
        log_fname = os.path.join(instance_folder, "out.txt")
        with open(log_fname, "w") as fh:
            fh.write(result.stdout)




if __name__ == "__main__":
    main()