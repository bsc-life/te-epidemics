import jsonschema
import argparse

DEFAULT_TE_PARAM = {
    "k": 1,
    "omega": 7,
    "s": 1,
    "delta": 4,
    "embedding": "uniform",
    "n_bins": 15,
    "symmetric_binning": True,
    "cases_cases": False
}

DEFAULT_SIR_PARAM = {
    "hub_idx":0,
    "t_max": 50,
    "i0": 10
}

def validate_config_parameters(config_dict):
    configSchema = {
        "type": "object",
        "properties": {
            "te_params": {"type": "object"},
            "sir_params": {"type": "object"}
        },
        "required": [
            "te_params", "sir_params"
            ]
    }
    
    jsonschema.validate(config_dict, configSchema)
    
    return True

def fill_missing_te_params(te_params):
    print("- Checking for missing TE params")
    for k,v in DEFAULT_TE_PARAM.items():
        if k in te_params:
            continue
        te_params[k] = v
        print(f"\t* Filling missing TE param \"{k}\" using default value \"{v}\"")
    return te_params

def fill_missing_sir_params(sir_params):
    print("- Checking for missing SIR model params")
    for k,v in DEFAULT_SIR_PARAM.items():
        if k in sir_params:
            continue
        sir_params[k] = v
        print(f"\t* Filling missing SIR model param \"{k}\" using default value \"{v}\"")
    return sir_params

def create_parser():

    parser = argparse.ArgumentParser(description="Calculate transfer entropy between MAR and cases.")

    parser.add_argument("--config", dest="config_json", action="store", required=True,
                        help="JSON file with config")

    parser.add_argument("--out", action="store", dest="output_folder", default=".",
                         help="Output folder to store results")

    parser.add_argument("--cpus", action="store", dest="cpus", default=None, type=int,
                         help="Nº of cpus use for parallel calculation")

    parser.add_argument("--save-ds", action="store", dest="save_ds", type=bool, default=True,
                            help="Run a single TE calculation for debuggin prouposes")


    parser.add_argument('--rnd-id', action='store_true', dest="rnd_id")

    parser.add_argument('--no-rnd-id', action='store_false', dest="rnd_id")
    
    parser.add_argument("--ds-format", action="store", dest="df_format", default="hdf5",
                            choices=("hdf5", "parquet", "csv"))

    return parser
