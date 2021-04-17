# PCPG Agent



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
from time import time
    

class PCPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network, self.optimizer, self.replay_buffer, self.density_model = dict(), dict(), dict(), dict()
        self.replay_buffer_actions = dict()
        self.replay_buffer_infos = dict()

        # create policy networks for explore, exploit and rollin phases
        for mode in ['explore', 'exploit', 'rollin']:
            self.network[mode] = config.network_fn()
            self.replay_buffer[mode] = []
            self.replay_buffer_actions[mode] = []
            self.replay_buffer_infos[mode] = []
            
        self.optimizer['explore'] = config.optimizer_fn(self.network['explore'].parameters())
        self.optimizer['exploit'] = config.optimizer_fn(self.network['exploit'].parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

        # list to store policies in the policy cover
        self.policy_mixture = [copy.deepcopy(self.network['explore'].state_dict())]

        # each policy will have its own optimizer
        self.policy_mixture_optimizers = [copy.deepcopy(self.optimizer['explore'].state_dict())]

        # weights among the policies in the cover, which is uses to sample
        self.policy_mixture_weights = torch.tensor([1.0])
        
        self.policy_mixture_returns = []
        self.timestamp = None

        # define exploration reward bonus
        if self.config.bonus == 'rnd':
            # RND bonus
            self.rnd_network = FCBody(self.config.state_dim).to(Config.DEVICE)
            self.rnd_pred_network = FCBody(self.config.state_dim).to(Config.DEVICE)
            self.rnd_optimizer = torch.optim.RMSprop(self.rnd_pred_network.parameters(), 0.001)
        elif self.config.bonus == 'randnet-kernel-s':
            # random network kernel mapping states to features
            if self.config.game == 'maze':
                self.kernel = ConvFCBodyMaze(size=config.maze_size, in_channels = 3, phi_dim = self.config.phi_dim).to(Config.DEVICE)
            else:
                self.kernel = FCBody(self.config.state_dim, hidden_units=(self.config.phi_dim, self.config.phi_dim)).to(Config.DEVICE)
        elif self.config.bonus == 'rbf-kernel':
            # RBF kernel
            self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.config.phi_dim)
            if isinstance(self.task.action_space, Box):
                self.rbf_feature.fit(X = np.random.randn(5, self.config.state_dim + self.config.action_dim))
            else:
                self.rbf_feature.fit(X = np.random.randn(5, self.config.state_dim + 1))

        if isinstance(self.task.action_space, Box):
            self.uniform_prob = self.continous_uniform_prob()
        else:
            self.uniform_prob = 1./self.config.action_dim

    # takes as input a minibatch of states (and possibly actions), returns exploration reward for each
    def compute_reward_bonus(self, states, actions = None):
        if self.config.bonus == 'rnd':
            states = torch.from_numpy(states).float().to(Config.DEVICE)
            rnd_target = self.rnd_network(states).detach()
            rnd_pred = self.rnd_pred_network(states).detach()
            rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(1)
            reward_bonus = rnd_loss.cpu().numpy()

        elif 'randnet-kernel' in self.config.bonus:
            phi = self.compute_kernel(tensor(states), actions)
            reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()
            
        elif 'rbf-kernel' in self.config.bonus:
            assert actions is not None
            phi = self.compute_kernel(tensor(states), tensor(actions))
            reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()
            
        elif 'id-kernel' in self.config.bonus:
            phi = self.compute_kernel(tensor(states), actions)
            reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()

            
        elif 'counts' in self.config.bonus: # can use ground truth counts in combolock for debugging
            reward_bonus = []
            for s in self.config.state_normalizer(states):
                s = tuple(s)
                if not s in self.density_model['explore'].keys():
                    cnts = 0
                else:
                    cnts = self.density_model['explore'][s]
                if self.config.bonus == 'counts':
                    reward_bonus.append(1.0/(1.0 + cnts))
                elif self.config.bonus == 'counts-sqrt':
                    reward_bonus.append(1.0/math.sqrt(1.0 + cnts))
                    
            reward_bonus = np.array(reward_bonus)
            
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
    # can specify whether to roll in using policy mixture, or instead use the latest policy
    def gather_trajectories(self, roll_in=True, add_bonus_reward=True, debug=False, mode=None, record_return=False):
        config = self.config
        states = self.states
        network = self.network[mode]

        roll_in_length = 0 if (debug or not roll_in) else random.randint(0, config.horizon - 1)
        roll_out_length = config.horizon - roll_in_length
        storage = Storage(roll_out_length)

        if roll_in_length > 0:
            assert roll_in
            # Sample previous policy to roll in
            i = torch.multinomial(self.policy_mixture_weights.cpu(), num_samples=1)
            self.network['rollin'].load_state_dict(self.policy_mixture[i])

            # Roll in
            for _ in range(roll_in_length):
                prediction = self.network['rollin'](states)
                next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
                if self.config.game == 'maze':
                    for i in info:
                        self.unique_pos.add(tuple(i['agent_pos']))            
                
                next_states = config.state_normalizer(next_states)
                states = next_states
                self.total_steps += config.num_workers

        # Roll out
        for i in range(roll_out_length):
            if i == 0 and roll_in: #if roll-in is false, then we ignore epsilon greedy and simply roll-out the current policy
                # we are using \hat{\pi}
                sample_eps_greedy = random.random() < self.config.eps
                if sample_eps_greedy:
                    if isinstance(self.task.action_space, Discrete):
                        actions = torch.randint(self.config.action_dim, (states.shape[0],)).to(Config.DEVICE)
                    elif isinstance(self.task.action_space, Box):
                        actions = self.uniform_sample_cont_random_acts(states.shape[0])
                    prediction = network(states, tensor(actions))
                else:
                    prediction = network(states)
                #update the log_prob_a by including the epsilon_greed
                prediction['log_pi_a'] = (prediction['log_pi_a'].exp() * (1.-self.config.eps) + self.config.eps*self.uniform_prob).log()
            else:
                # we are using \pi
                prediction = network(states)

            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))

            if self.config.game == 'maze':
                for i in info:
                    self.unique_pos.add(tuple(i['agent_pos']))            

            if add_bonus_reward:
                s = config.state_normalizer(states)
                reward_bonus = self.config.reward_bonus_normalizer(self.compute_reward_bonus(s,to_np(prediction['a'])))
                rewards = self.config.bonus_coeff*self.config.horizon*reward_bonus
                assert(all(rewards >= 0))

            if record_return:
                self.record_online_return(info)

            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'i': list(info), 
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers


#        assert(np.array(terminals).all()) # debug
        self.states = states
        prediction = network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(roll_out_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        return storage

    def log(self, s):
        logtxt(self.logger.log_dir + '.txt', s, show=True, date=False)


    # compute the mapping from states (and possibly actions) to features
    def compute_kernel(self, states, actions = None):
        actions_one_hot = tensor(np.eye(self.config.action_dim)[actions])
#        state_actions = torch.cat((tensor(states).to(Config.DEVICE), actions_one_hot), dim=1)
        
        if self.config.bonus == 'randnet-kernel-s':
            phi = F.normalize(self.kernel(tensor(states).to(Config.DEVICE)), p=2, dim=1)
        elif self.config.bonus == 'randnet-kernel-sa':
            phi = F.normalize(self.kernel(state_actions), p=2, dim=1)
        elif self.config.bonus == 'id-kernel-s':
            phi = states.to(Config.DEVICE)
        elif self.config.bonus == 'id-kernel-sa':
            phi = state_actions
        elif self.config.bonus == 'rbf-kernel':
            assert actions is not None
            if actions is None:
                phi = self.rbf_feature.transform(states.cpu().numpy())
                phi = torch.tensor(phi).to(Config.DEVICE)
            else:
                #concatenate state and action features
                np_states = states.cpu().numpy()
                np_actions = actions.cpu().numpy()
                if isinstance(self.task.action_space, Discrete):
                    np_actions = np.expand_dims(np_actions, axis = 1)
                assert np_actions.ndim == 2 and np_actions.shape[0] == np_states.shape[0] 
                states_acts_cat = np.concatenate((np_states, self.clip_actions(np_actions)), axis = 1)
                phi = self.rbf_feature.transform(states_acts_cat)
                phi = torch.tensor(phi).to(Config.DEVICE)
        else:
            raise NotImplementedError
        return phi


    # for visualizing visitations in combolock
    def log_visitations(self, visitations):
        self.log('lock1')
        self.log(np.around(visitations[0], 3))
        self.log('lock2')
        self.log(np.around(visitations[1], 3))

    # turn count-based density model into visitation table
    def compute_state_visitations(self, density_model, use_one_hot=False):
        locks = [np.zeros((3, self.config.horizon-1)), np.zeros((3, self.config.horizon-1))]
        N = sum(list(density_model.values()))
        for state in density_model.keys():
            if use_one_hot:
                k = np.argmax(state)
                (s, l, h) = np.unravel_index(k , (3, 3, self.config.horizon))
                if l in [0, 1]:
                    locks[l][s][h] += float(density_model[state]) / N
            else:
                if not all(np.array(state)==0.0):
                    s = np.argmax(state[:3])
                    l = int(state[-1])
                    h = np.argmax(state[3:-1])
                    locks[l][s][h] += float(density_model[state]) / N
        return locks
    
        
    # update the density model using data from replay buffer.
    # also computes covariance matrices for kernel case. 
    def update_density_model(self, mode=None):
        replay_buffer = self.replay_buffer[mode]
        replay_buffer_act = self.replay_buffer_actions[mode]
        states = torch.cat(sum(replay_buffer, []))
        actions = torch.cat(sum(replay_buffer_act,[]))
        
        if self.config.bonus == 'rnd':
            states = states.to(Config.DEVICE)
            targets = self.rnd_network(states).detach()
            data = DataLoader(TensorDataset(states, targets), batch_size = 100, shuffle=True)

            for i in range(1):
                total_loss = 0
                losses = []
                for j, batch in enumerate(data):
                    self.rnd_optimizer.zero_grad()
                    pred = self.rnd_pred_network(batch[0])
                    loss = F.mse_loss(pred, batch[1], reduction='none')
                    (loss.mean()).backward()
                    self.rnd_optimizer.step()
                    total_loss += loss.mean().item()
                    losses.append(loss)
                print(f'[RND loss: {total_loss / j:.5f}]')
            bonuses = torch.cat(losses).view(-1)
        
        elif self.config.bonus == 'rbf-kernel':
            N = states.shape[0]
            ind = np.random.choice(N, min(2000, N), replace=False)
            pdists = scipy.spatial.distance.pdist((states.cpu().numpy())[ind])
            self.rbf_feature.gamma = 1./(np.median(pdists)**2)
            phi = self.compute_kernel(states, actions = actions)
            n, d = phi.shape
            sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).to(Config.DEVICE)
            self.density_model = torch.inverse(sigma).detach()

            covariance_matrices = []
            assert len(replay_buffer) == len(replay_buffer_act)
            for i in range(len(replay_buffer)):
                states = torch.cat(replay_buffer[i])
                actions = torch.cat(replay_buffer_act[i])
                phi = self.compute_kernel(states,actions)
                n, d = phi.shape
                sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).to(Config.DEVICE)
                covariance_matrices.append(sigma.detach())
            m = 0
            for matrix in covariance_matrices:
                m = max(m, matrix.max())
            covariance_matrices = [matrix / m for matrix in covariance_matrices]

        elif 'kernel' in self.config.bonus:
            N = states.shape[0]
            phi = self.compute_kernel(states, actions)
            n, d = phi.shape
            sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).to(Config.DEVICE)
            self.density_model = torch.inverse(sigma).detach()

            covariance_matrices = []
            assert len(replay_buffer) == len(replay_buffer_act)
            for i in range(len(replay_buffer)):
                states = torch.cat(replay_buffer[i])
                actions = torch.cat(replay_buffer_act[i])
                phi = self.compute_kernel(states, actions)
                n, d = phi.shape
                sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).to(Config.DEVICE)
                covariance_matrices.append(sigma.detach().cpu())
            m = 0
            for matrix in covariance_matrices:
                m = max(m, matrix.max())
            covariance_matrices = [matrix / m for matrix in covariance_matrices]

            
        
        elif 'counts' in self.config.bonus:
            states = [tuple(s) for s in states.numpy()]
            unique_states = list(set(states))
            self.density_model[mode] = dict(zip(unique_states, [0] * len(unique_states)))
            for s in states: self.density_model[mode][s] += 1
            bonuses = torch.tensor([1.0/self.density_model[mode][s] for s in states])
            covariance_matrices, visitations = [], []
            for i, states in enumerate(replay_buffer):
                states = [tuple(s) for s in torch.cat(states).numpy()]
                density_model = dict(zip(unique_states, [0] * len(unique_states)))
                for s in states: density_model[s] += 1
                sums=torch.tensor([density_model[s] for s in unique_states]).float()
                covariance_matrices.append(torch.diag(sums) + torch.eye(len(unique_states)))
                visitations.append(self.compute_state_visitations(density_model))

            m = 0
            for matrix in covariance_matrices:
                m = max(m, matrix.max())
            covariance_matrices = [matrix / m for matrix in covariance_matrices]

        if mode == 'explore': self.optimize_policy_mixture_weights(covariance_matrices)

        # for combolock, compute the visitations for each policy
        if 'combolock' in self.config.game:

            visitations = []
            states = torch.cat(sum(replay_buffer, []))
            states = [tuple(s) for s in states.numpy()]
            unique_states = list(set(states))
            for i, states in enumerate(self.replay_buffer[mode]):
                states = [tuple(s) for s in torch.cat(states).numpy()]
                density_model = dict(zip(unique_states, [0] * len(unique_states)))
                for s in states: density_model[s] += 1
#                visitations.append(self.compute_state_visitations(self.replay_buffer_infos[mode][i]))
                visitations.append(self.compute_state_visitations(density_model))
                
            if mode == 'explore':
                weighted_visitations = [np.zeros((3, self.config.horizon - 1)), np.zeros((3, self.config.horizon - 1))]
                for i in range(len(visitations)):
                    weighted_visitations[0] += self.policy_mixture_weights[i].item()*visitations[i][0]
                    weighted_visitations[1] += self.policy_mixture_weights[i].item()*visitations[i][1]

                for i in range(len(visitations)):
                    self.log(f'\nstate visitations for policy {i}:')
                    self.log_visitations(visitations[i])

                self.log(f'\nstate visitations for weighted policy mixture:')
                self.log_visitations(weighted_visitations)
                
            elif mode == 'exploit':
                self.log(f'\nstate visitations for exploit policy:')
                self.log_visitations(visitations[-1])

        self.reward_bonus_normalizer= RescaleNormalizer()



    # optimize policy mixture weights using log-determinant loss
    def optimize_policy_mixture_weights(self, covariance_matrices):
        d = covariance_matrices[0].shape[0]
        N = len(covariance_matrices)
        if N == 1:
            self.policy_mixture_weights = torch.tensor([1.0])
        else:
            self.log_alphas = nn.Parameter(torch.randn(N))
            opt = torch.optim.Adam([self.log_alphas], lr=0.001)
            for i in range(5000):
                opt.zero_grad()
                sigma_weighted_sum = torch.zeros(d, d)
                for n in range(N):
                    sigma_weighted_sum += F.softmax(self.log_alphas, dim=0)[n]*covariance_matrices[n]
                loss = -torch.logdet(sigma_weighted_sum)
                if math.isnan(loss.item()):
                    pdb.set_trace()
                if not i % 500:
                    print(f'optimizing log det, loss={loss.item()}')
                loss.backward()
                opt.step()
            with torch.no_grad():
                self.policy_mixture_weights = F.softmax(self.log_alphas, dim=0)
        self.log(f'\npolicy mixture weights: {self.policy_mixture_weights.numpy()}')


    # roll out using explore/exploit policies and store data in replay buffer
    def update_replay_buffer(self):
        print('[gathering trajectories for replay buffer]')
        for mode in ['explore', 'exploit']:
            states, actions, returns, infos = [], [], [], []
            for _ in range(self.config.n_rollouts_for_density_est):
                new_traj = self.gather_trajectories(roll_in=False, add_bonus_reward=False, mode=mode,
                                                    record_return=(mode=='exploit'))            
                states += new_traj.cat(['s'])
                returns += new_traj.cat(['r'])
                actions += new_traj.cat(['a']) #append actions as well
                infos += new_traj.i

            mean_return = torch.cat(returns).cpu().mean()*self.config.horizon
            if mode == 'explore':
                self.policy_mixture_returns.append(mean_return.item())
                self.log(f'[policy mixture returns: {np.around(self.policy_mixture_returns, 3)}]')
            states = [s.cpu() for s in states]
            print(f'return ({mode}): {mean_return}')
            self.replay_buffer[mode].append(states)
 
            actions = [a.cpu() for a in actions]
            self.replay_buffer_actions[mode].append(actions)
            self.replay_buffer_infos[mode].append(sum(infos, []))
        
    # optimize explore and/or exploit policies
    def optimize_policy(self):            
        for mode in ['explore', 'exploit']:
            if mode == 'exploit' and self.epoch < self.config.start_exploit:
                continue
            for i in range(self.config.n_policy_loops):
                rewards = self.step_optimize_policy(mode=mode)
                if not i % 5: print(f'[optimizing policy ({mode}), step {i}, mean return: {rewards.mean():.5f}]')

        self.policy_mixture.append(copy.deepcopy(self.network['explore'].state_dict()))
        self.policy_mixture_optimizers.append(copy.deepcopy(self.optimizer['explore'].state_dict()))
        print(f'{len(self.policy_mixture)} policies in mixture')

        
    def initialize_new_policy(self, mode):
        self.network[mode] = self.config.network_fn()
        self.optimizer[mode] = self.config.optimizer_fn(self.network[mode].parameters())
        

    # gather a batch of data and perform some policy optimization steps
    def step_optimize_policy(self, mode=None):
        config = self.config
        network = self.network[mode]
        optimizer = self.optimizer[mode]

        states, actions, rewards, log_probs_old, returns, advantages = [], [], [], [], [], []
        self.time('reset')

        # gather the trajectories
        for i in range(self.config.n_traj_per_loop):

            #some fraction of the time, we roll-in from the policy itself (so no data is wasted), half of the time from mixture
            coin = np.random.rand()
            if coin <= (1.0-self.config.proll): #simply roll-in with the policy itself, not from mixture:
                traj = self.gather_trajectories(add_bonus_reward=(mode=='explore'), mode=mode, roll_in = False)
            else: #from mixture
                traj = self.gather_trajectories(add_bonus_reward=(mode=='explore'), mode=mode, roll_in = True)

            states += traj.cat(['s'])
            actions += traj.cat(['a'])
            log_probs_old += traj.cat(['log_pi_a'])
            returns += traj.cat(['ret'])
            rewards += traj.cat(['r'])
            advantages += traj.cat(['adv'])
#        self.time('gathering trajectories')
        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        log_probs_old = torch.cat(log_probs_old, 0)
        returns = torch.cat(returns, 0)
        rewards = torch.cat(rewards, 0)
        advantages = torch.cat(advantages, 0)
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == advantages.shape[0] == returns.shape[0]
        
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.time('reset')

        # optimize the policy using the gathered trajectories using PPO objective
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = network(sampled_states, sampled_actions)

                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
                optimizer.step()
#        self.time('optimizing policy')

        return rewards.mean()


    # we clip the actions since the policy uses Gaussian distribution to sample actions
    # in the continuous case. This avoids the policy generating large actions to maximize
    # the negative log-det.
    def clip_actions(self, actions): 
        #action: numpy
        if isinstance(self.task.action_space, Box):
            #only clip in continuos setting. 
            for i in range(self.config.action_dim):
                actions[:, i] = np.clip(actions[:,i], self.task.action_space.low[i], 
                    self.task.action_space.high[i])
        return actions
        
    
    def eval_step(self, state):
        network = self.network['exploit']
        prediction = network(state)
        action = to_np(prediction['a'])
        return action


    #test function for policy: 
    def test_exploit_policy_performance(self):
        network = self.network['exploit']
        roll_in_length = self.config.horizon
        storage = Storage(roll_in_length)
        num_trajs = 0
        total_rews = 0
        states = self.task.reset() #reset environment, so roll-in from the beignning
        for i in range(roll_in_length):
            prediction = network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            num_trajs += terminals.sum()
            total_rews += rewards.sum()
        
        assert num_trajs > 0
        return total_rews / num_trajs #this may overestimates rewards...but fair for all baselines as well..
