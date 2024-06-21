W=14
D=7
E="round"
CPUS=8
X="risk_ij"
Y="new_cases_by_100k"

LAYERS="TE_cnig_provincias TE_zon_bas_13 TE_abs_09"
for l in $LAYERS;
do
    echo "Runnin on layer ${l} with parameters w:${W} d:${D} e:${E} x:${X} y:${Y}"
    python scripts/calculate_transfer_entropy.py -w ${W} -d ${D} -e ${E} -c ${CPUS} -x ${X} -y ${Y} outputs/${l}
    echo "============== DONE ===============\n"
done
