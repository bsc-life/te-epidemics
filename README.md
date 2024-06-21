## Create a Virtual Environment and install python dependencies

```virtualenv -p python3 venv
source activate venv
pip install -r requirements.txt
```

## Running Analysis for Spain

Download all the datasets from Zenodo:
[https://zenodo.org/api/records/12207212/files-archive]([url](https://zenodo.org/api/records/12207212/files-archive))

unzip the file and move all files into the data folder.

Run the following command to set up Spain data:

```bash scripts/setup_experiments.sh```

## Run SIR simulations and TE

Run the following command to run the Transfer Entropy Analysis on Spain data:

```bash scripts/run_te_experiments.sh```

Run the following command to set up and run SIR simulations:

```bash run_SIR_simulations.sh```

Run the following command to run the Transfer Entropy Analysis on the simulated epidemic
```bash scripts/run_te_on_SIRs.sh```
