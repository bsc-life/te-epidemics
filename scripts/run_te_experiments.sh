W=28
D=14
E="standard"
CPUS=8
X="risk_ij"
Y="new_cases_by_100k"
LAYERS="TE_cnig_provincias TE_zon_bas_13 TE_abs_09"
USE_ETE="true"
RUN_CMD="python scripts/calculate_transfer_entropy.py"
for l in $LAYERS;
do
    echo
    echo "============= TE on ${l} ==========="
    CMD_PARAMS="-w ${W} -d ${D} -e ${E} -c ${CPUS} -x ${X} -y ${Y} --use_ete ${USE_ETE} "
    echo "* Runnin on layer ${l} with parameters:"
    echo "${CMD_PARAMS}" | sed 's/ \-/\n\-/g'
    # ${RUN_CMD} ${CMD_PARAMS}  outputs/${l}
    echo "============== DONE ==============="
done
