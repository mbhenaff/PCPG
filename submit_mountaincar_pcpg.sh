#!/bin/bash

for env in MountainCarContinuous-v0; do
for alg in ppo-pcpg2; do
for lr in 5e-4; do
for bonus_coeff in 0.1; do
for n_policy_loops in 100; do
for n_traj_per_loop in 50; do
for bonus in rbf-kernel; do
for phi_dim in 10; do
for proll in 0.5; do 
for ridge in 0.01; do 
for seed in 1 2 3 4 5; do
	python run.py -env $env -alg $alg -bonus_coeff $bonus_coeff -bonus $bonus -proll $proll \
	     -ridge $ridge -n_policy_loops $n_policy_loops -n_traj_per_loop $n_traj_per_loop \
	     -phi_dim $phi_dim -seed $seed -lr $lr 

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
done
