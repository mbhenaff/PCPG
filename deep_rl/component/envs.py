#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import os
import gym
import numpy as np
import torch
import json, pdb, random
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from ..utils import *

try:
    import roboschool
except ImportError:
    pass

#import baselines




class DiabolicalLockMaze(object):

    def __init__(self, horizon, seed=1, noise='bernoulli'):
        self.horizon = horizon
        self.n_states = 3
        self.num_actions = 10
        self.n_locks = 2
        self.locks = [build_env_homer(horizon=horizon-1, seed=seed), build_env_homer(horizon=horizon-1, seed=seed*10)]
        self.optimal_reward = 5.0
        self.suboptimal_reward = 2.0
        if seed % 2 == 0:
            self.locks[0].env.optimal_reward = self.optimal_reward
            self.locks[1].env.optimal_reward = self.suboptimal_reward
        else:
            self.locks[0].env.optimal_reward = self.suboptimal_reward
            self.locks[1].env.optimal_reward = self.optimal_reward

        self.reward_range = (0, self.optimal_reward)
        self.metadata = None
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.n_features = self.locks[0].observation_space.shape[0] + 1
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_features,), dtype=np.float)
                
    def reset(self):
        self.h = 0
        obs = np.zeros(self.observation_space.shape)
        return obs

    def seed(self, seed):
        pass

    def step(self, action):
        assert self.n_locks == 2
        if self.h == 0:
            # initial state, action will choose lock
            if action < 5:
                self.lock = self.locks[0]
                self.lock_id = 0
            else:
                self.lock = self.locks[1]
                self.lock_id = 1
            obs = self.lock.reset()
            reward = 0.0
            done = False
            info = {'state': (0, self.h, self.lock_id)}
        else:
            obs, reward, done, info = self.lock.step(action)
            info['state'] = info['state'] + (self.lock_id,)

        self.h += 1
        obs = np.append(obs, self.lock_id)
        return obs, reward, done, info








def build_env_homer(horizon=10, seed=1):
    import homer_envs
    config = homer_envs.default_config.default_config()
    config['horizon'] = horizon
    config['noise'] = 'none'
    config['save_trace'] = False
    config['seed'] = seed
    homer_envs.environment_wrapper.GenerateEnvironmentWrapper.adapt_config_to_domain('diabcombolock', config)
    env = homer_envs.environment_wrapper.GenerateEnvironmentWrapper('diabcombolock', config)
    return env


def build_env_maze(horizon=200, size=20):
    import maze
    env = maze.MazeEnv(size=size, time=200, holes=0, num_goal=1)
    return env

def build_env_combolock(horizon=10):
    # constant file contains hyperparameters for the model and learning algorithm.
    with open("baselines/homer/data/diabcombolock/config.json") as f:
        config = json.load(f)
        config["horizon"] = horizon
        config["noise"] = 'hadamhardg'
        envwrap.GenerateEnvironmentWrapper.adapt_config_to_domain('diabcombolock', config)
    print(json.dumps(config, indent=2))
    envwrap.GenerateEnvironmentWrapper.adapt_config_to_domain('diabcombolock', config)
    env = envwrap.GenerateEnvironmentWrapper('diabcombolock', config)
    env.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(config["obs_dim"],),dtype=np.float)
    env.action_space = gym.spaces.Discrete(10)
    env.reward_range = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)
    env.metadata = None
    return env



# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(env_id, seed, rank, episode_life=True, horizon=4, noise='bernoulli', dimension=30):
    def _thunk():
        random_seed(seed)
        if 'diabcombolockhallway' == env_id:
            env = DiabolicalLockMaze(horizon=horizon, seed=seed, noise=noise)
        elif 'maze' == env_id:
            env = build_env_maze(horizon=100, size=20)            
        elif env_id.startswith("dm"):
            import dm_control2gym
            _, domain, task = env_id.split('-')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        if not 'combolock' in env_id:
            env.seed(seed + rank)
        env = OriginalReturnWrapper(env) #aggregate episode return
        if is_atari:
            env = wrap_deepmind(env,
                                episode_life=episode_life,
                                clip_rewards=False,
                                frame_stack=False,
                                scale=False)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3:
                env = TransposeImage(env)
            env = FrameStack(env, 4)

        return env

    return _thunk


class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            try:
                info['num_rooms'] = len(info['episode'])
            except:
                pass
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
            info['num_rooms'] = None

        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


# The original one in baselines is really bad
class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        data = []
        for i in range(self.num_envs):
            obs, rew, done, info = self.envs[i].step(self.actions[i])
            if done:
                obs = self.envs[i].reset()
            data.append([obs, rew, done, info])
        obs, rew, done, info = zip(*data)
        return obs, np.asarray(rew), np.asarray(done), info

    def reset(self):
        return [env.reset() for env in self.envs]

    def close(self):
        return


class Task:
    def __init__(self,
                 name,
                 num_envs=1,
                 single_process=True,
                 seed = 1, 
                 log_dir=None,
                 episode_life=True,
                 horizon=10,
                 noise='bernoulli', 
                 dimension=30):
        if log_dir is not None:
            mkdir(log_dir)
        if 'diabcombolockhallway' in name:
            env = make_env(name, seed, 0, episode_life, horizon=horizon, noise=noise)
            envs = [env for i in range(num_envs)]
        elif 'maze' == name:
            env = make_env(name, seed, 0, episode_life, horizon=horizon)
            envs = [env for i in range(num_envs)]            
        else:
            envs = [make_env(name, seed, i, episode_life) for i in range(num_envs)]
        if single_process:
            Wrapper = DummyVecEnv
        else:
            Wrapper = SubprocVecEnv
        self.env = Wrapper(envs)
        self.name = name
        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.env.observation_space.shape))

        self.action_space = self.env.action_space
        if isinstance(self.action_space, Discrete):
            self.action_dim = self.action_space.n
        elif isinstance(self.action_space, Box):
            self.action_dim = self.action_space.shape[0]
        else:
            assert 'unknown action space'

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        if isinstance(self.action_space, Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return self.env.step(actions)


if __name__ == '__main__':
    task = Task('Hopper-v2', 5, single_process=False)
    state = task.reset()
    while True:
        action = np.random.rand(task.observation_space.shape[0])
        next_state, reward, done, _ = task.step(action)
        print(done)
