#!/bin/bash

for env in diabcombolockhallway; do
for horizon in 6; do 
for alg in ppo-rnd; do
for lr in 1e-3 5e-4 1e-4; do
for bonus_coeff in 1000.0; do
for seed in {1..1}; do
	python run.py -env $env -alg $alg -bonus_coeff $bonus_coeff -horizon $horizon \
	       -seed $seed -lr $lr 

done
done
done
done
done
done
