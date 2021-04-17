#!/bin/bash

DIR=/private/home/mikaelhenaff/projects/Policy-Gradient-with-Exploration/code/DeepRL/
SWEEP_NAME=rnd_mountaincar
JOBSCRIPTS=scripts
mkdir -p ${JOBSCRIPTS}
SAVE_ROOT=/checkpoint/mikaelhenaff/${SWEEP_NAME}/
mkdir -p stdout stderr
queue=learnfair

for env in MountainCarContinuous-v0; do
for alg in ppo-rnd; do
for lr in 1e-3 5e-4 1e-4; do
for bonus_coeff in 1.0 10.0 100.0 1000.0 10000.0; do
for seed in {1..5}; do
        SAVE=${SAVE_ROOT}/${SWEEP_NAME}
        mkdir -p ${SAVE}
        JNAME=${SWEEP_NAME}.results.${alg}.env_${env}_horizon_${horizon}_lr_${lr}_bcoeff_${bonus_coeff}.seed_${seed}
        SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
        SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm
        extra=""
        echo "#!/bin/sh" > ${SCRIPT}
	echo "#!/bin/sh" > ${SLURM}
        echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
        echo "#SBATCH --output=stdout/${JNAME}.%j" >> ${SLURM}
        echo "#SBATCH --error=stderr/${JNAME}.%j" >> ${SLURM}
        echo "#SBATCH --partition=$queue" >> ${SLURM}
        echo "#SBATCH --signal=USR1@120" >> ${SLURM}
        echo "#SBATCH --gres=gpu:1" >> ${SLURM}
        echo "#SBATCH --mem=100000" >> ${SLURM}
        echo "#SBATCH --time=4320" >> ${SLURM}
        echo "#SBATCH --nodes=1" >> ${SLURM}
        echo "#SBATCH --cpus-per-task=1" >> ${SLURM}
        echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
        echo "srun sh ${SCRIPT}" >> ${SLURM}
        echo "echo \$SLURM_JOB_ID >> jobs" >> ${SCRIPT}
        echo "{ " >> ${SCRIPT}
        echo "nvidia-smi" >> ${SCRIPT}
        echo "cd $DIR" >> ${SCRIPT}
	echo source activate my_pytorch_env >> ${SCRIPT}
	echo python run.py -env $env -alg $alg -bonus_coeff $bonus_coeff \
	     -seed $seed -lr $lr -log_dir "${SAVE}" >> ${SCRIPT}

        echo "nvidia-smi" >> ${SCRIPT}
        echo "kill -9 \$\$" >> ${SCRIPT}
        echo "} & " >> ${SCRIPT}
        echo "child_pid=\$!" >> ${SCRIPT}
        echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
        echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
        echo "while true; do     sleep 1; done" >> ${SCRIPT}
        sbatch ${SLURM}
done
done
done
done
done
