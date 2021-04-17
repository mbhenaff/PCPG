# PC-PG Algorithm 


## Running the code

This repo contained the PyTorch implementation of the PCPG algorith, described in the following paper: [PCPG: Policy Cover Directed Exploration for Provable Policy Gradient Learning](https://arxiv.org/abs/2007.08459), by [Alekh Agarwal](http://alekhagarwal.net/), [Mikael Henaff](http://mikaelhenaff.net), [Sham Kakade](https://homes.cs.washington.edu/~sham/) and [Wen Sun](https://wensun.github.io/).

The necessary dependencies are in the ```environment.yml``` file. You can run

```
conda env create -f environment.yml
./setup.sh
```

To run PCPG on the Bidirectional Diabolical Combination Lock environment, execute:


```
python run.py -alg ppo-pcpg -env diabcombolockhallway -horizon 6 -lr 0.0005
```

The scripts ```submit_combolock_pcpg.sh```, ```submit_combolock_ppo_rnd.sh``` and ```submit_combolock_ppo.sh``` will the PC-PG, PPO+RND and vanilla PPO algorithms on the combination locks for different horizon lengths. PC-PG solves the task for all horizon lengths at least 95% of the time, while PPO+RND only succeeds roughly 50% of the time due to not adequately exploring both locks. Vanilla PPO fails due to the antishaped rewards. 


## Visualizing the policy cover

For the Diabolical Combination Lock environment, the learned policies in the policy cover can be seen in the .txt files written to the output folder. For example, for horizon length 5, the policies and mixture weights are given by:

```
policy mixture weights: [0.36501506 0.40568325 0.22930169]

state visitations for policy 0:
lock1
[[0.048 0.004 0.    0.    0.   ]
 [0.033 0.004 0.    0.    0.   ]
 [0.    0.073 0.08  0.081 0.081]]
lock2
[[0.051 0.004 0.    0.    0.   ]
 [0.034 0.005 0.    0.    0.   ]
 [0.    0.077 0.085 0.086 0.086]]

state visitations for policy 1:
lock1
[[0.069 0.076 0.061 0.064 0.06 ]
 [0.056 0.049 0.064 0.061 0.065]
 [0.    0.    0.    0.    0.   ]]
lock2
[[0.022 0.019 0.02  0.02  0.02 ]
 [0.019 0.022 0.021 0.018 0.018]
 [0.    0.    0.    0.003 0.004]]

state visitations for policy 2:
lock1
[[0.003 0.003 0.004 0.004 0.003]
 [0.004 0.003 0.003 0.003 0.004]
 [0.    0.    0.    0.    0.   ]]
lock2
[[0.077 0.093 0.085 0.075 0.075]
 [0.083 0.067 0.075 0.085 0.066]
 [0.    0.    0.    0.    0.019]]

state visitations for weighted policy mixture:
lock1
[[0.046 0.033 0.026 0.027 0.025]
 [0.036 0.022 0.027 0.025 0.027]
 [0.    0.027 0.029 0.03  0.03 ]]
lock2
[[0.045 0.031 0.028 0.025 0.025]
 [0.039 0.026 0.026 0.027 0.022]
 [0.    0.028 0.031 0.033 0.037]]
```

The scripts ```submit_mountaincar_pcpg.sh``` and ```submit_mountaincar_ppo_rnd.sh``` will run the algorithms on the MountainCar environment.


## Acknowledgements

This codebase is based on the excellent [repo](https://github.com/ShangtongZhang/DeepRL) of Shangtong Zhang