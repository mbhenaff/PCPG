#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import argparse, os
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
#from IPython import embed

import torch
torch.set_default_tensor_type(torch.FloatTensor)


# PPO and PPO+RND
def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.log_interval = 1000
    config.num_workers = 100

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, single_process=True, seed=config.seed, horizon = config.horizon)
    config.eval_env = Task(config.game, seed=config.seed, horizon=config.horizon)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.lr)
    config.state_dim = config.eval_env.state_dim
    
    base_network = FCBody(config.state_dim)

    
    if config.alg != 'ppo':
        if config.norm_rew_b == 1:
            config.reward_bonus_normalizer = MeanStdNormalizer()
        else:
            config.reward_bonus_normalizer = RescaleNormalizer()
        
    agent.logger = Logger(None, config.log_dir + '/tensorboard/')

    
    if isinstance(config.task_fn().action_space, Box):
        config.network_fn = lambda: GaussianActorCriticNet(config.state_dim, config.action_dim, base_network)
    elif isinstance(config.task_fn().action_space, Discrete):
        config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, base_network)
    

    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.optimization_epochs = 5
    config.gradient_clip = 0.5
    config.rollout_length = config.horizon
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.max_steps = 3e6*config.horizon
    run_steps(PPOAgent(config))

def ppo_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    
    agent.logger = Logger(None, config.log_dir + '/tensorboard/')
    
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 8
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)


    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = int(2e7)
    run_steps(PPOAgent(config))
    


def pcpg_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.log_interval = 1000
    config.reward_bonus_normalizer = RescaleNormalizer()


    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, single_process=True, seed=config.seed, horizon = config.horizon)
    config.eval_env = Task(config.game, seed=config.seed, horizon=config.horizon)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.lr)
    config.eval_episodes = 100 #eval over 100 trajectories
    config.state_dim = config.eval_env.state_dim

    config.num_workers = 50
    base_network = FCBody(config.state_dim)
    config.optimization_epochs = 5
    config.mini_batch_size = 32 * 5
    
    

    if isinstance(config.task_fn().action_space, Box):
        config.network_fn = lambda: GaussianActorCriticNet(config.state_dim, config.action_dim, base_network)
    elif isinstance(config.task_fn().action_space, Discrete):
        config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, base_network)
    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = config.horizon
    config.ppo_ratio_clip = 0.2
    config.max_steps = 10e9*config.horizon
    
    if 'combolock' in config.game:
        config.start_exploit = config.horizon 
        config.max_epochs = 3*config.horizon
        config.rmax = 5.0
        config.n_rollouts_for_density_est = 50
    else:
        config.max_epochs = 50
        if config.game == 'MountainCarContinuous-v0':
            config.start_exploit = 0
            config.rmax = 100 # hardcoding for now
        config.n_rollouts_for_density_est = 10

    
    if config.alg == 'ppo-pcpg':
        run_steps(PCPGAgent(config))
    elif config.alg == 'ppo-pcpg2':
        run_steps(PCPG2Agent(config))



if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    select_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=0, help='which GPU to use')
    parser.add_argument('-log_dir', type=str, default='./results/', help='where to write results')
    parser.add_argument('-env', type=str, default='diabcombolockhallway', help='diabcombolockhallway | MountainCarContinuous-v0')
    parser.add_argument('-horizon', type=int, default=5, help='horizon for combolock')
    parser.add_argument('-alg', type=str, default='ppo-pcpg', help='ppo-pcpg | ppo-rnd | ppo')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-bonus_coeff', type=float, default=1.0, help='coefficient for intrinsic reward')
    # PCPG hyperparameters
    parser.add_argument('-bonus', type=str, default='id-kernel-s', help='kernel for PCPG (\phi in the paper)')
    parser.add_argument('-eps', type=float, default=0.05)
    parser.add_argument('-proll', type=float, default=1.0)
    parser.add_argument('-n_policy_loops', type=int, default=100, help='how many policy optimization steps for each epoch')
    parser.add_argument('-n_traj_per_loop', type=int, default=50)
    parser.add_argument('-norm_rew', type=int, default=0)
    parser.add_argument('-norm_rew_b', type=int, default=0)
    parser.add_argument('-phi_dim', type=int, default=64)
    parser.add_argument('-ridge', type=float, default=0.01)
    parser.add_argument('-system', type=str, default='gcr')
    config = parser.parse_args()
    select_device(config.device)

    os.system(f'mkdir -p {config.log_dir}')
    random_seed(config.seed)
    if config.alg == 'ppo':
        if config.env == 'MontezumaRevengeNoFrameskip-v4':
            ppo_pixel(game=config.env,
                        lr=config.lr,
                        seed=config.seed,
                        log_dir = config.log_dir,
                        rnd = 0,
                        alg='ppo',
                        system = config.system)
        else:
            ppo_feature(game=config.env,
                        lr=config.lr,
                        horizon=config.horizon,
                        seed=config.seed,
                        log_dir = config.log_dir,
                        rnd = 0,
                        alg='ppo',
                        system = config.system)        
    elif config.alg == 'ppo-rnd':
        ppo_feature(game=config.env,
                    lr=config.lr,
                    horizon=config.horizon,
                    seed=config.seed,
                    rnd = 1,
                    phi_dim = config.phi_dim, 
                    rnd_bonus = config.bonus_coeff,
                    alg='ppo-rnd',
                    log_dir = config.log_dir,
                    norm_rew=config.norm_rew,
                    norm_rew_b=config.norm_rew_b,
                    system = config.system)
    elif config.alg in ['ppo-pcpg', 'ppo-pcpg2']:
        pcpg_feature(game=config.env,
                    lr=config.lr,
                    ridge=config.ridge,
                    horizon=config.horizon,
                    seed=config.seed,
                    eps = config.eps,
                    proll = config.proll,
                    bonus = config.bonus, 
                    bonus_coeff = config.bonus_coeff,
                    phi_dim = config.phi_dim, 
                    n_policy_loops = config.n_policy_loops,
                    n_traj_per_loop = config.n_traj_per_loop,
                    alg=config.alg,
                    log_dir = config.log_dir,
                    system = config.system)
