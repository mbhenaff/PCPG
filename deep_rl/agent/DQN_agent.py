#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import pdb

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        multi_env = not self.config.num_workers == 1
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            q_values = self._network(config.state_normalizer(self._state))
        if multi_env:
            q_values = to_np(q_values)
        else:
            q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            if multi_env:
                action = [np.random.randint(0, q_values.shape[1]) for i in range(q_values.shape[0])]
            else:
                action = np.random.randint(0, len(q_values))
        else:
            if multi_env:
                action = np.argmax(q_values, axis=1)
            else:
                action = np.argmax(q_values)
        if multi_env:
            next_state, reward, done, info = self._task.step(action)
#            entry = [self._state, action, reward, next_state, done, info]
            entry = [self._state, action, reward, next_state, done, info]
            self._total_steps += len(action)
#            self._total_steps += q_values.shape[0]
        else:
            next_state, reward, done, info = self._task.step([action])
            entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
            self._total_steps += 1
        self._state = next_state
        return entry


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)


        if config.rnd == 1:
            n_hidden = 500
            self.rnd_network = nn.Sequential(nn.Linear(config.state_dim, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden)).cuda()

            self.rnd_pred_network = nn.Sequential(nn.Linear(config.state_dim, n_hidden),
                                                  nn.ReLU(),
                                                  nn.Linear(n_hidden, n_hidden),
                                                  nn.ReLU(),
                                                  nn.Linear(n_hidden, n_hidden)).cuda()
            self.rnd_optimizer = config.optimizer_fn(self.rnd_pred_network.parameters())
        

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        multi_env = not config.num_workers == 1
        self.total_steps += config.num_workers
        if multi_env:
            transitions_ = []
            for i in range(config.num_workers):
                transition_ = [transitions[0][j][i] for j in range(6)]
                transitions_.append(transition_)
            transitions = transitions_

#        self.total_steps += self.config.num_workers
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences

            if multi_env:
                states = self.config.state_normalizer(states).squeeze()
                next_states = self.config.state_normalizer(next_states).squeeze()
                q_next = self.target_network(next_states).detach()
            else:
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = self.target_network(next_states).detach()
                
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]


            if multi_env:
                terminals = tensor(terminals).squeeze()
                rewards = tensor(rewards).squeeze()
            else:
                terminals = tensor(terminals)
                rewards = tensor(rewards)


            if config.rnd == 1:
                self.rnd_optimizer.zero_grad()
                s = torch.from_numpy(config.state_normalizer(states)).cuda().float()
                rnd_target = self.rnd_network(s).detach()
                rnd_pred = self.rnd_pred_network(s)
                rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(1)
                (rnd_loss.mean()).backward()
                self.rnd_optimizer.step()
                rewards += config.rnd_bonus*rnd_loss.detach()
                
                
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
