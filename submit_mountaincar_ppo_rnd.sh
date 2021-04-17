#!/bin/bash

for env in MountainCarContinuous-v0; do
for alg in ppo-rnd; do
for lr in 1e-3 5e-4 1e-4; do
for bonus_coeff in 1.0 10.0 100.0 1000.0 10000.0; do
for seed in {1..5}; do
    python run.py -env $env -alg $alg -bonus_coeff $bonus_coeff \
	   -seed $seed -lr $lr 

done
done
done
done
done
