#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

import math, random, pdb, copy, sys
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import scipy
from sklearn.kernel_approximation import RBFSampler
from IPython import embed
from time import time
from sklearn.decomposition import PCA

class RPG3Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network, self.optimizer = dict(), dict()
        self.density_model = np.zeros((self.config.phi_dim, self.config.phi_dim))

        assert self.config.bonus == 'randnet-kernel'
        self.covariance_matrices = []

        for mode in ['explore', 'exploit', 'rollin']:
            self.network[mode] = config.network_fn()

        #set up optimizer: one for explore and one for exploit policy
        self.optimizer['explore'] = config.optimizer_fn(self.network['explore'].parameters())
        self.optimizer['exploit'] = config.optimizer_fn2(self.network['exploit'].parameters())

        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

        #policy mixture setup
        self.policy_mixture = [copy.deepcopy(self.network['explore'].state_dict())]
        self.policy_mixture_weights = torch.tensor([1.0])
        #self.policy_mixture_returns = []
        self.timestamp = None

        if self.config.bonus == 'randnet-kernel' and self.config.image_input ==0:
            self.kernel = FCBody(self.config.state_dim, hidden_units=(self.config.phi_dim, self.config.phi_dim)).to(Config.DEVICE)
        elif self.config.bonus=='randnet-kernel' and self.config.image_input==1:
            print("construct random network...")
            self.kernel = testConvBodywithAction(4, self.config.phi_dim, self.config.action_dim, input_action = False).to(Config.DEVICE)

        self.uniform_prob = 1./self.config.action_dim
        self.max_T = self.config.horizon
        self.step_prob_dist = [(1.-self.config.discount)*self.config.discount**h for h in range(self.max_T)]
        print('max_T: {}'.format(self.max_T))

        #self.max_T =  self.config.horizon #min(int(2./(1.-self.config.discount)), 1000, self.config.horizon)

    # takes as input a minibatch of states, returns exploration reward
    def compute_reward_bonus(self, states, actions = None):
        assert actions is not None
        phi = self.compute_kernel(states, actions)
        reward_bonus = np.diag(phi.dot(self.density_model).dot(phi.transpose()))

        return reward_bonus

    def time(self, tag=''):
        if self.time is None or tag=='reset':
            self.timestamp = time()
        else:
            t = time()
            print(f'{tag} took {t - self.timestamp:.4f}s')
            self.timestamp = t

    # gather trajectories following a policy and return them in a buffer.
    # explore mode uses exploration bonus as reward, exploit uses environment reward
    # can specify whether to roll in using policy mixture or not

    def sample_time_step(self):
        h = np.random.choice(self.max_T, 1, self.step_prob_dist)[0]
        #h = np.random.choice(self.max_T, 1)[0]
        #print(h)
        return h

    def gather_trajectories(self, roll_in=True, debug=False, mode=None, record_return=False, add_reward_bonus = True):
        config = self.config
        #states = self.states
        states = config.state_normalizer(self.task.reset())

        assert isinstance(states, np.ndarray) is True
        network = self.network[mode]

        roll_in_length = 0 if (debug or not roll_in) else self.sample_time_step()
        roll_out_length =  self.config.horizon - roll_in_length # self.max_T#min(self.max_T - roll_in_length+1, int(2./(1.-self.config.discount))) #1/(1-lambda), 
        storage = Storage(roll_out_length)

        if roll_in_length > 0:
            assert roll_in
            # Sample previous policy to roll in
            i = torch.multinomial(self.policy_mixture_weights.cpu(), num_samples=1)
            self.network['rollin'].load_state_dict(self.policy_mixture[i])

            # Roll in
            for _ in range(roll_in_length):
                prediction = self.network['rollin'](tensor(states))
                next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
                next_states = config.state_normalizer(next_states)
                states = next_states
                assert isinstance(states, np.ndarray) is True
                self.total_steps += config.num_workers

        # Roll out
        for i in range(roll_out_length):
            if i == 0 and roll_in: #if roll-in is false, then we ignore epsilon greedy and simply roll-out the current policy
                # we are using \hat{\pi}
                sample_eps_greedy = random.random() < self.config.eps
                if sample_eps_greedy:
                    actions = torch.randint(self.config.action_dim, (states.shape[0],)).to(Config.DEVICE)
                    prediction = network(tensor(states), tensor(actions))
                else:
                    prediction = network(tensor(states))
                #update the log_prob_a by including the epsilon_greed
                prediction['log_pi_a'] = (prediction['log_pi_a'].exp() * (1.-self.config.eps) + self.config.eps*self.uniform_prob).log()
            else:
                # we are using \pi
                prediction = network(tensor(states))

            next_states, ext_rewards, terminals, info = self.task.step(to_np(prediction['a']))

            int_rewards = self.compute_reward_bonus(states, to_np(prediction['a']))
            normalized_int_rews = self.config.reward_bonus_normalizer(np.copy(int_rewards))
            if mode == 'explore':
                rewards = int_rewards#normalized_int_rews #int_rewards/self.config.reward_bonus_normalizer.rms.var**0.5
            elif mode == 'exploit':
                rewards = config.reward_normalizer(ext_rewards) + int_rewards/self.config.reward_bonus_normalizer.rms.var**0.5

            self.record_online_return(info)

            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({
                'r': np.expand_dims(rewards, axis= -1),
                'm': np.expand_dims(1-terminals, axis = -1),
                'i': list(info),
                'ext_r': np.expand_dims(ext_rewards, axis=-1),
                'int_r': np.expand_dims(int_rewards, axis=-1),
                's': states})
            states = next_states
            assert isinstance(states, np.ndarray) is True
            self.total_steps += config.num_workers

        #assert(np.array(terminals).all()) # debug
        self.states = states
        prediction = network(tensor(states))
        storage.add(prediction)
        storage.placeholder()

        advantages = np.zeros((config.num_workers, 1))
        returns = prediction['v'].detach().cpu().numpy()
        for i in reversed(range(roll_out_length)):
            m_i = storage.m[i] if mode == 'exploit' else 1. #if mode = explore, i.e., pure explore mode, we treat it as non-episodic
            returns = storage.r[i] + config.discount * m_i * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach().cpu().numpy()
            else:
                td_error = storage.r[i] + config.discount * m_i * storage.v[i + 1].detach().cpu().numpy() - storage.v[i].detach().cpu().numpy()
                advantages = advantages * config.gae_tau * config.discount * m_i + td_error

            assert isinstance(advantages, np.ndarray) is True and isinstance(returns, np.ndarray) is True
            storage.adv[i] = advantages
            storage.ret[i] = returns

        return storage

    def log(self, s):
        logtxt(self.logger.log_dir + '.txt', s, show=True, date=False)

    def compute_kernel(self, states, acts = None):
        if isinstance(self.task.action_space, Discrete):
            actions= np.eye(self.config.action_dim)[acts]
        elif isinstance(self.task.action_space, Box):
            actions = acts
        assert states.shape[0] == actions.shape[0] and actions.ndim == 2
        N = states.shape[0]
        phi = np.zeros((N, self.config.phi_dim))
        t = 0
        while True:
            tnext = t + 512
            if tnext <= N:
                end_pos = tnext
            else:
                end_pos = N
            phi[t:end_pos] = (F.normalize(self.kernel(tensor(states[t:end_pos]),tensor(self.clip_actions(actions[t:end_pos]))), p=2,dim=1)).detach().cpu().numpy()
            #phi[t:end_pos] = self.kernel(tensor(states[t:end_pos]),tensor(self.clip_actions(actions[t:end_pos]))).detach().cpu().numpy()
            if end_pos == N:
                break
            else:
                t = tnext
        return phi

    #update the sigma inverse
    def compute_covariance_inverse(self, use_log_det = True):
        print('compute inverse of total_cov and the mixture weights')
        N = len(self.covariance_matrices)
        assert N > 0

        self.optimize_policy_mixture_weights(self.covariance_matrices)
        weights = self.policy_mixture_weights.detach().cpu().numpy()
        assert weights.shape[0] == N

        d = self.covariance_matrices[0].shape[0]
        total_cov = 1e-7*np.eye(d)
        for i in range(N):
            if use_log_det is True:
                total_cov += self.covariance_matrices[i] * (N * weights[i])
            else:
                total_cov += self.covariance_matrices[i]

        self.density_model = np.linalg.inv((total_cov + total_cov.transpose())/2.)

    # optimize policy mixture weights using log-determinant loss
    def optimize_policy_mixture_weights(self, covariance_matrices):
        print("update the inverse of the curr covariance matrix")
        d = covariance_matrices[0].shape[0]
        N = len(covariance_matrices)
        if N == 1:
            self.policy_mixture_weights = torch.tensor([1.0])
        else:
            self.log_alphas = nn.Parameter(torch.randn(N))
            opt = torch.optim.Adam([self.log_alphas], lr=0.0001)
            for i in range(5000):
                opt.zero_grad()
                sigma_weighted_sum = torch.zeros(d, d)
                for n in range(N):
                    sigma_weighted_sum += F.softmax(self.log_alphas, dim=0)[n]*torch.tensor(covariance_matrices[n])
                loss = -torch.logdet(sigma_weighted_sum)
                if math.isnan(loss.item()):
                    pdb.set_trace()
                if not i % 1500:
                    print(f'optimizing log det, loss={loss.item()}')
                loss.backward()
                opt.step()
            with torch.no_grad():
                self.policy_mixture_weights = F.softmax(self.log_alphas, dim=0)
        self.log(f'\npolicy mixture weights: {self.policy_mixture_weights.numpy()}')


    def update_covariance_matrices(self):

        mode = 'explore'
        print('[gathering trajectories from the latest explore policy to estimate sigma]')
        states, actions, rewards, ext_rewards, int_rewards, infos = [], [], [], [], [], [] 

        for _ in range(self.config.n_rollouts_for_density_est):
            coin = np.random.rand()
            if coin < 1.: #0.5:
                new_traj = self.gather_trajectories(roll_in=False, mode=mode)
            else:
                new_traj = self.gather_trajectories(roll_in=True, mode=mode)

            states += new_traj.cat(['s'], ndarray =True)
            rewards += new_traj.cat(['r'], ndarray=True)
            ext_rewards += new_traj.cat(['ext_r'], ndarray=True)
            int_rewards += new_traj.cat(['int_r'], ndarray=True)
            actions += new_traj.cat(['a'], ndarray=False) #append actions as well, actions are in tensor cuda
            infos += new_traj.i

        mean_rewards = np.array(rewards).mean()
        mean_ext_rewards = np.array(ext_rewards).mean()
        mean_int_rewards = np.array(int_rewards).mean()
        states = np.concatenate(states,0)
        actions = np.concatenate(actions, 0)
        max_num_rooms_sofar = max([max([env_info['num_rooms_sofar'] for env_info in info]) for info in infos])

        #self.policy_mixture_returns.append(mean_return.item())
        #self.log(f'[policy mixture returns: {np.around(self.policy_mixture_returns, 3)}]')
        print(f'return ({mode}): {mean_rewards, mean_ext_rewards, mean_int_rewards, max_num_rooms_sofar}')
        #phi = (F.normalize(self.kernel(tensor(states),tensor(self.clip_actions(actions))), p=2,dim=1)).detach().cpu().numpy()
        phi = self.compute_kernel(states, actions)
        sigma = phi.transpose().dot(phi)/phi.shape[0]
        self.covariance_matrices.append((sigma+sigma.transpose())/2. + 1e-7*np.eye(phi.shape[1]))

    # optimize explore and/or exploit policies
    def optimize_policy(self):
        if self.config.init_new_policy == 1:
            self.initialize_new_policy('explore')

        for mode in ['explore', 'exploit']:
            if mode == 'exploit' and self.epoch < self.config.start_exploit:
                continue
            for i in range(self.config.n_policy_loops):
                ext_rewards, int_rewards, rewards, max_room_num_curr_loop = self.step_optimize_policy(mode=mode)
                if not i % 5: print(f'[optimizing policy ({mode}), step {i}, mean return: {ext_rewards.mean():.5f}, {int_rewards.mean():.5f}, {rewards.mean():.5f},{max_room_num_curr_loop:.5f}]')

        self.policy_mixture.append(copy.deepcopy(self.network['explore'].state_dict()))
        print(f'{len(self.policy_mixture)} policies in mixture')

    def initialize_new_policy(self, mode):
        self.network[mode] = self.config.network_fn()
        self.optimizer[mode] = self.config.optimizer_fn(self.network[mode].parameters())

    # gather a batch of data and perform some policy optimization steps
    def step_optimize_policy(self, mode=None):
        config = self.config
        network = self.network[mode]
        optimizer = self.optimizer[mode]

        states, actions, rewards, int_rewards, ext_rewards, log_probs_old, returns, advantages = [], [], [], [], [], [],[],[]
        infos = []
        self.time('reset')
        for i in range(self.config.n_traj_per_loop):

            #half of the time, we roll-in from the policy itself (so no data is wasted), half of the time from mixture
            coin = np.random.rand()
            if coin <= 0.5: #simply roll-in with the policy itself, not from mixture:
                traj = self.gather_trajectories(mode = mode, roll_in = False)
            else: #from mixture
                traj = self.gather_trajectories(mode = mode, roll_in = True)

            states += traj.cat(['s'], ndarray=True)
            actions += (traj.cat(['a'],ndarray=False)) #.detach().cpu().numpy()
            log_probs_old += traj.cat(['log_pi_a'],ndarray=False) #.detach().cpu().numpy()
            returns += traj.cat(['ret'],ndarray=True)
            rewards += traj.cat(['r'], ndarray=True)
            int_rewards += traj.cat(['int_r'], ndarray=True)
            ext_rewards += traj.cat(['ext_r'], ndarray=True)
            advantages += traj.cat(['adv'], ndarray=True)
            infos += traj.i

        states = np.concatenate(states,0)
        actions = np.concatenate(actions, 0)
        log_probs_old = np.concatenate(log_probs_old, 0)
        returns = np.concatenate(returns, 0)
        ext_rewards = np.concatenate(ext_rewards, 0)
        int_rewards = np.concatenate(int_rewards, 0)
        rewards = np.concatenate(rewards, 0)
        advantages = np.concatenate(advantages, 0)

        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == advantages.shape[0] == returns.shape[0]
        #advantages = (advantages - advantages.mean()) / advantages.std()
        assert isinstance(states, np.ndarray) is True and isinstance(actions, np.ndarray) is True and isinstance(advantages, np.ndarray) is True

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.shape[0]), config.mini_batch_size)
            for batch_indices in sampler:
                #batch_indices = batch_indices
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]
                prediction = network(tensor(sampled_states), tensor(sampled_actions))

                ratio = (prediction['log_pi_a'] - tensor(sampled_log_probs_old)).exp()
                obj = ratio * tensor(sampled_advantages)
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * tensor(sampled_advantages)
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (tensor(sampled_returns) - prediction['v']).pow(2).mean()

                optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
                optimizer.step()

        max_num_rooms_sofar = max([max([env_info['num_rooms_sofar'] for env_info in info]) for info in infos])
        return ext_rewards.mean(), int_rewards.mean(), rewards.mean(), max_num_rooms_sofar



    #we clip the actions since the policy uses Gaussian distribution to sample actions
    #this avoids the policy generating large actions to maximum the neg log-det.
    def clip_actions(self, actions): 
        #action: numpy
        if isinstance(self.task.action_space, Box):
            #only clip in continuos setting. 
            for i in range(self.config.action_dim):
                actions[:, i] = np.clip(actions[:,i], self.task.action_space.low[i], 
                    self.task.action_space.high[i])
        else:
            #embed()
            return np.eye(self.config.action_dim)[actions] if actions.ndim == 1 else actions
        return actions

    def eval_step(self, state):
        network = self.network['exploit']
        prediction = network(self.config.state_normalizer(tensor(np.expand_dims(state[0],axis=0))))

        action = to_np(prediction['a'])
        return action

