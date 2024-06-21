BASE_OUTPUT="outputs"
METAPOPs=("chain" "ring" "star")

Ws=(9 9 8)
Ds=(4 5 5)
E="round"
X="risk_ij"
Y="new_cases_by_100k"

CPUS=8

for i in $(seq 0 2)
do
    M=${METAPOPs[$i]}
    W=${Ws[$i]}
    D=${Ds[$i]}
    EXPERIMENT_FOLDER="${BASE_OUTPUT}/SIR_simulation_${M}"
 
    # TE{i,j} R{i,j}(t) --> I{j}(t)
    echo "Runnin on SIR ${M} with parameters w:${W} d:${D} e:${E} x:${X} y:${Y}"
    python scripts/calculate_transfer_entropy.py -w ${W} -d ${D} -e ${E} -c ${CPUS} -x ${X} -y ${Y} ${EXPERIMENT_FOLDER}

    # TE{i,j} I{i}(t) --> I{j}(t)
    echo "Runnin on SIR ${M} with parameters w:${W} d:${D} e:${E} x:${X} y:${Y}"
    python scripts/calculate_transfer_entropy.py -w ${W} -d ${D} -e ${E} -c ${CPUS} -x ${Y} -y ${Y} ${EXPERIMENT_FOLDER}

    echo "============== DONE ==============="
    echo
done
