#!/usr/bin/bash

# Basic config
OUTPUT_FOLDER="outputs"
DATA_FOLDER="data"
CPUS=8

# Metapopulations parameters
REGIONS=9
POP=10000
DELTA=0.05

# Parameters for SIR simulations
DT=0.1
T_MAX=180
I0=10
R0=2
MU=0.3333
Z0_IDXS=(0 0 1)
REPS=1000


METAPOPS=("chain" "ring" "star")

for i in $(seq 0 2)
do
    M=${METAPOPS[$i]}
    Z0=${Z0_IDXS[$i]}
    EPICOMUTE_ARGS="--dt ${DT} --tmax ${T_MAX} --i0 ${I0} --r0 ${R0} --mu ${MU} --z0 ${Z0} -n ${REPS}"
    METAPOP_FNAME="${DATA_FOLDER}/mobility_${M}_${REGIONS}_pop.npy"
    echo "Creating metapopulationc with the following parameters"
    echo "-t ${M} -n ${REGIONS} -p ${POP} -d ${DELTA} ${DATA_FOLDER}"
    python scripts/generate_mobility_network.py -t ${M} -n ${REGIONS} -p ${POP} -d ${DELTA} ${DATA_FOLDER}
    echo
    echo "Running SIR simulations using:"
    echo ${EPICOMUTE_ARGS}
    python scripts/simulate_SIR.py -c ${CPUS} ${EPICOMUTE_ARGS} ${OUTPUT_FOLDER} ${METAPOP_FNAME}
    echo "================= DONE =================="
    echo
done
