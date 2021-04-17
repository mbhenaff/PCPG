#!/bin/bash

for env in diabcombolockhallway; do
for horizon in 6 11 16; do 
for alg in ppo-pcpg; do
for lr in 5e-4; do
for bonus_coeff in 1.0; do
for n_policy_loops in 100; do
for n_traj_per_loop in 50; do
for ridge in 0.01; do 
for proll in 1.0; do 
for seed in {1..20}; do
	python run.py -env $env -alg $alg -bonus_coeff $bonus_coeff -proll $proll -horizon $horizon \
	     -ridge $ridge -n_policy_loops $n_policy_loops -n_traj_per_loop $n_traj_per_loop \
	     -seed $seed -lr $lr -log_dir './results/'
done    
done
done
done
done
done
done
done
done
done
