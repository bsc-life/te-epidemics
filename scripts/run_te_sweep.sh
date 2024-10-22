LAYER="TE_cnig_provincias"
E="standard"
X="risk_ij"
Y="new_cases_by_100k"
W=28
D=14
CPUS=8
USE_ETE="true"
RUN_CMD="python scripts/calculate_transfer_entropy.py"
for D in $(seq 1 21);
do
    echo
    echo "============= TE on ${l} ==========="
    CMD_PARAMS="-w ${W} -d ${D} -e ${E} -c ${CPUS} -x ${X} -y ${Y} --use_ete ${USE_ETE} "
    echo "* Runnin on layer ${LAYER} with parameters:"
    echo "${CMD_PARAMS}" | sed 's/ \-/\n\-/g'
    ${RUN_CMD} ${CMD_PARAMS}  outputs/${LAYER}
    echo "============== DONE ==============="
done
