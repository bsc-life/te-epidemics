NUMTHREADS=8

LAYER="SIR_simulation_star"
OUTPUT=results/SIR_R2_M1_3_N1000
DATE=`date +"%Y%m%d"`

LOGFILE="param_sweep_${LAYER}_${DATE}.log"
LOGFILE="${OUTPUT}/${LOGFILE}"
touch $LOGFILE

I0=10
R0=2
MU=0.33
TMAX=300
N=1000
HUB=0
ODM=data/SIR_mobility_matrices/mobility_star_9.npy

echo "* Running SIR simulation using: -i0 ${I0} -r0 ${R0} -mu ${MU} -tmax ${TMAX} -n ${N} -hub ${HUB} ${OUTPUT} ${ODM}"
echo "* Running SIR simulation using: -i0 ${I0} -r0 ${R0} -mu ${MU} -tmax ${TMAX} -n ${N} -hub ${HUB} ${OUTPUT} ${ODM}" >> ${LOGFILE}
python scripts/calculate_SIR_model.py -i0 ${I0} -r0 ${R0} -mu ${MU} -tmax ${TMAX} -n ${N} -hub ${HUB} -c${NUMTHREADS} ${OUTPUT} ${ODM}

echo "============= DONE! ==============="
echo "============= DONE! ===============" >> ${LOGFILE}

OUTPUT="${OUTPUT}/${LAYER}"
EMBEDDING="round"
echo "Running parameter sweep for ${LAYER} (${OUTPUT})"
for W in `seq 3 1 30`
do
    for D in `seq 1 1 7`
    do  
        CMD="scripts/calculate_transfer_entropy.py ${OUTPUT} -d${D} -w${W} -e${EMBEDDING} -c${NUMTHREADS}"
        echo "Calculating TE Using delta=${D} omega=${W}"
        echo "Calculating TE Using delta=${D} omega=${W}" >> ${LOGFILE}
	echo ${CMD} >> ${LOGFILE}
        python $CMD >> ${LOGFILE}
        echo "============ END RUN ============" >> ${LOGFILE}
    done
done
