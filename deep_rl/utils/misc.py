#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime, math
import torch
import time
import pdb, pickle, os, copy
from .torch_utils import *
from pathlib import Path

def logtxt(fname, s, date=True, show=False):
    if show: print(s)
    if not os.path.isdir(os.path.dirname(fname)):
        os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    if date:
        f.write(f'{str(datetime.datetime.now())}: {s}\n')
    else:
        f.write(f'{s}\n')
    f.close()


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    agent.epoch = 0


    test_performance = []
    best_exploit_performance = -math.inf
    best_exploit_params = None
    
    logtxt(agent.logger.log_dir + '.csv', 'episodes,mean episode reward', date=False)
    while True:
        total_episodes = agent.total_steps / config.horizon
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))


        if 'ppo-pcpg' in config.alg:
            agent.log(f'\n###### EPOCH {agent.epoch} #####')
            if config.alg == 'ppo-pcpg':
                if agent.epoch == agent.config.start_exploit:
                    agent.initialize_new_policy('exploit')
                agent.update_replay_buffer()
                agent.update_density_model(mode='explore')
            elif config.alg == 'ppo-pcpg2':
                agent.update_replay_buffer()
                agent.update_density_model(mode='explore-exploit')
            agent.optimize_policy()
            agent.epoch += 1

            if agent.config.game == 'maze':
                # reward-free exploration, measure number of unique positions visited
                avg_episodic_return = len(agent.unique_pos)
            else:
                avg_episodic_return = agent.eval_episodes()['episodic_return_test']
            test_performance.append(avg_episodic_return)
            if avg_episodic_return > best_exploit_performance:
                best_exploit_performance = avg_episodic_return
                best_exploit_params = copy.deepcopy(agent.network['exploit'].state_dict())
                
            print("#### at epoch {}, avg episodic return is {}".format(agent.epoch, avg_episodic_return))
            print(test_performance)

            if agent.epoch == agent.config.max_epochs:
                logtxt(agent.logger.log_dir + '.txt', f'reverting to best policy with performance {best_exploit_performance:.4f}', show=True)
                agent.network['exploit'].load_state_dict(best_exploit_params)
                total_episodes = agent.total_steps / config.horizon
                n_eval_episodes = config.num_workers*agent.config.n_rollouts_for_density_est
                running_mean_reward = agent.eval_episodes(n_episodes = n_eval_episodes)['episodic_return_test']
                logtxt(agent.logger.log_dir + '.txt', f'final performance: {running_mean_reward}', date=False)            
                agent.close()
                
                logtxt(agent.logger.log_dir + '.txt', f'{total_episodes + n_eval_episodes},{running_mean_reward}', date=False, show=True)            
                logtxt(agent.logger.log_dir + '.csv', f'{total_episodes + n_eval_episodes},{running_mean_reward}', date=False)
                torch.save(agent.replay_buffer, agent.logger.log_dir + '.buffer')
                break
            else:
                logtxt(agent.logger.log_dir + '.txt', f'{total_episodes},{avg_episodic_return}', date=False)            
                logtxt(agent.logger.log_dir + '.csv', f'{total_episodes},{avg_episodic_return}', date=False)

        # for non-PCPG algorithms
        else:
            agent.step()
            if agent.total_steps > 0 and config.log_interval and not total_episodes % config.log_interval and ('ppo-pcpg' not in config.alg) and len(agent.ep_rewards) > 0:
                
                running_mean_reward_10_ep = np.mean(agent.ep_rewards[-10:])
                running_mean_reward_1000_ep = np.mean(agent.ep_rewards[-1000:])
                log_string = 'steps %d, episodes %d, %.2f steps/s, total rew %.2f, mean rew (10 ep) %.4f, mean rew (1000 ep) %.4f' % (agent.total_steps, total_episodes, config.log_interval / (time.time() - t0), agent.cumulative_reward, running_mean_reward_10_ep, running_mean_reward_1000_ep)
                agent.logger.info(log_string)
                logtxt(agent.logger.log_dir + '.txt', log_string)
                t0 = time.time()
                if config.game == 'maze':
                    logtxt(agent.logger.log_dir + '.csv', f'{total_episodes}, {len(agent.unique_pos)}', date=False)
                else:
                    logtxt(agent.logger.log_dir + '.csv', f'{total_episodes}, {running_mean_reward_1000_ep}', date=False)
            if config.eval_interval and not agent.total_steps % config.eval_interval:
                agent.eval_episodes()
            if config.max_steps and agent.total_steps >= config.max_steps:
                agent.close()
                break

        
        
        agent.switch_task()

#    save_dir = agent.logger.log_dir + '/traces/'
#    os.system(f'mkdir -p {save_dir}')
#    torch.save(agent.traces, save_dir + '/traces.pth')

        


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def generate_tag(params):
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    log_dir = params['log_dir']
    del params['game']
    del params['run']
    del params['log_dir']
    str = ['%s_%s' % (k, v) for k, v in sorted(params.items())]
    tag = '%s-%s-run-%d' % (game, '-'.join(str), run)
    params['tag'] = tag
    params['game'] = game
    params['run'] = run
    params['log_dir'] = log_dir


def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
