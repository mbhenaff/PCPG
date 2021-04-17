import os, pdb
import time
import pickle
import numpy as np

from homer_envs.noise_gen import generated_hadamhard_matrix


class Environment(object):
    """
    An environment skeleton. Defaults to simple MAB
    H = 1, K=2, rewards are bernoulli, sampled from dirichlet([1,1]) prior.
    """

    BERNOULLI, GAUSSIAN, HADAMHARD, HADAMHARDG, NONE = range(5)

    def __init__(self):
        self.state = None
        self.h = 0

    def start_episode(self):
        self.h = 0
        self.state = self.start()
        return self.make_obs(self.state), {"state": None if self.state is None else tuple(self.state)}

    def get_actions(self):
        if self.state is None:
            raise Exception("Episode not started")
        if self.h == self.horizon:
            return None
        return self.actions
    
    def make_obs(self, s):
        return s

    def act(self, a):
        if self.state is None:
            raise Exception("Episode not started")

        if self.h == self.horizon:
            new_state = None
        else:
            new_state = self.transition(self.state, a)
            self.h += 1

        r = self.reward(self.state, a, new_state)
        self.state = new_state

        # Create a dictionary containing useful debugging information
        info = {"state": None if self.state is None else tuple(self.state)}

        return self.make_obs(self.state), r, info

    def get_num_actions(self):
        return len(self.actions)

    def is_tabular(self):
        return True

    def get_dimension(self):
        assert not self.is_tabular(), "Not a featurized environment"
        return self.dim


class DiabolicalCombinationLock(Environment):

    def __init__(self, horizon, swap, num_actions, anti_shaping_reward, noise_type, optimal_reward=1.0, anti_shaping_reward2=1.0, seed=1):
        """
        :param horizon: Horizon of the MDP
        :param swap: Probability for stochastic edges
        :param noisy_dim: Dimension of noise
        :param noise_type: Type of Noise
        """

        Environment.__init__(self)
        self.horizon = horizon
        self.swap = swap
        self.noise_type = noise_type
        self.num_actions = num_actions
        self.optimal_reward = optimal_reward
        self.optimal_reward_prob = 1.0
        self.rng = np.random.RandomState(seed)

        assert anti_shaping_reward < self.optimal_reward * self.optimal_reward_prob, \
            "Anti shaping reward shouldn't exceed optimal reward which is %r" % \
            (self.optimal_reward * self.optimal_reward_prob)
        self.anti_shaping_reward = anti_shaping_reward
        self.anti_shaping_reward2 = anti_shaping_reward2

        assert num_actions >= 2, "Atleast two actions are needed"
        self.actions = list(range(0, num_actions))

        self.opt_a = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)
        self.opt_b = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)

        if noise_type == Environment.GAUSSIAN:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            self.dim = self.horizon + 4

        elif noise_type == Environment.BERNOULLI or noise_type == Environment.NONE:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            self.dim = self.horizon + 4 + self.horizon  # Add noise of length horizon
            

        elif noise_type == Environment.HADAMHARD:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        elif noise_type == Environment.HADAMHARDG:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

    def transition(self, x, a):

        if x is None:
            raise Exception("Not in any state")

        b = self.rng.binomial(1, self.swap)

        if x[0] == 0 and a == self.opt_a[x[1]]:
            if b == 0:
                return [0, x[1] + 1]
            else:
                return [1, x[1] + 1]
        if x[0] == 1 and a == self.opt_b[x[1]]:
            if b == 0:
                return [1, x[1] + 1]
            else:
                return [0, x[1] + 1]
        else:
            return [2, x[1] + 1]

    def make_obs(self, x):

        if x is None or self.dim is None:
            return x
        else:

            if self.noise_type == Environment.NONE:

                v = np.zeros(self.dim, dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
            
            elif self.noise_type == Environment.BERNOULLI:

                v = np.zeros(self.dim, dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                # TODO
#                v[self.horizon + 4:] = self.rng.binomial(1, 0.5, self.horizon)

            elif self.noise_type == Environment.GAUSSIAN:

                v = np.zeros(self.dim, dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)

            elif self.noise_type == Environment.HADAMHARD:

                v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                v = np.matmul(self.hadamhard_matrix, v)

            elif self.noise_type == Environment.HADAMHARDG:

                v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)
                v = np.matmul(self.hadamhard_matrix, v)

            else:
                raise AssertionError("Unhandled noise type %r" % self.noise_type)

            return v

    def start(self):

        # Start stochastically in one of the two live states
        toss_value = self.rng.binomial(1, 0.5)

        if toss_value == 0:
            return [0, 0]
        elif toss_value == 1:
            return [1, 0]
        else:
            raise AssertionError("Toss value can only be 1 or 0. Found %r" % toss_value)

    def reward(self, x, a, next_x):

        # If the agent reaches the final live states then give it the optimal reward.
        if (x == [0, self.horizon-1] and a == self.opt_a[x[1]]) or (x == [1, self.horizon-1] and a == self.opt_b[x[1]]):
            return self.optimal_reward * self.rng.binomial(1, self.optimal_reward_prob)

        # If reaching the dead state for the first time then give it a small anti-shaping reward.
        # This anti-shaping reward is anti-correlated with the optimal reward.
        if x is not None and next_x is not None:
            if x[0] != 2 and next_x[0] == 2:
                return self.anti_shaping_reward * self.rng.binomial(1, 0.5)
            elif x[0] != 2 and next_x[0] != 2:
                return -self.anti_shaping_reward2 / (self.horizon-1)
        

        return 0

    def get_optimal_value(self):
        return self.optimal_reward * self.optimal_reward_prob

    def is_tabular(self):
        return self.dim is None

    def save(self, folder_name):
        """ Save the environment given the folder name """

        timestamp = time.time()

        if not os.path.exists(folder_name + "/env_%d" % timestamp):
            os.makedirs(folder_name + "/env_%d" % timestamp, exist_ok=True)

        with open(folder_name + "/env_%d/diabcombolock" % timestamp, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(env_folder_name):
        """ Load the environment from the environment folder name """

        with open(env_folder_name + "/diabcombolock", "rb") as f:
            env = pickle.load(f)

        return env






class RandomGridWorld(Environment):
    """
    A M x M grid with walls and a trembling hand. 
    Horizon is always 2 M
    """
    def __init__(self, M, swap=0.1, dim=2, noise=0.0):
        Environment.__init__(self)
        self.M = M
        self.swap = swap
        self.noise = noise
        self.dim = dim
        self.seed = 147
        np.random.seed(self.seed)
        self.goal = None

        self.maze = self.generate_maze(self.M)
        self.state = None

        self.actions = [(0,1), (0,-1), (1,0), (-1,0)]
        print("ENV: Generated Random Grid World")
        print("Size: %dx%d, Start: [%d,%d], Goal: [%d,%d], H: %d, Cells: %d" %
              (self.M, self.M, 0, 0, self.goal[1], self.goal[0], self.horizon, np.count_nonzero(self.maze)))
        self.print_maze()

    def is_tabular(self):
        return False

    def generate_maze(self, M):
        """ 
        Adapted from http://code.activestate.com/recipes/578356-random-maze-generator/
        """
        mx = M; my = M
        maze = np.matrix(np.zeros((mx,my)))
        dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
        stack = [(0,0)]

        while len(stack) > 0:
            (cx, cy) = stack[-1]
            maze[cy,cx] = 1
            # find a new cell to add
            nlst = [] # list of available neighbors
            for i in range(4):
                nx = cx + dx[i]; ny = cy + dy[i]
                if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                    if maze[ny,nx] == 0:
                        # of occupied neighbors must be 1
                        ctr = 0
                        for j in range(4):
                            ex = nx + dx[j]; ey = ny + dy[j]
                            if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                                if maze[ey,ex] == 1: ctr += 1
                        if ctr == 1: nlst.append(i)
            # if 1 or more neighbors available then randomly select one and move
            if len(nlst) > 0:
                ir = nlst[np.random.randint(0, len(nlst))]
                cx += dx[ir]; cy += dy[ir]
                stack.append((cx, cy))
            elif self.goal is None:
                self.horizon = 2*len(stack)
                (gx,gy) = stack.pop()
                self.goal = [gx,gy]
            else:
                stack.pop()

        return(maze)

    def make_obs(self,s):
        if s is None:
            return None
        tmp = s.copy()
        v = np.random.normal(0, self.noise, self.dim)
        if self.dim > 2:
            mult = np.random.choice(range(self.M), size=self.dim - 2)
            tmp.extend(mult)
        return tmp+v

    def start(self):
        return [0,0]

    def reward(self,x,a):
        if x == self.goal:
            return 1
        return 0

    def transition(self, x, a):
        if x == self.goal:
            return None
        nx = x[0]+a[0]
        ny = x[1]+a[1]
        if nx < 0 or nx >= self.M or ny < 0 or ny >= self.M:
            ## Cannot go off the grid
            return x
        if self.maze[ny,nx] == 0:
            ## Cannot enter a wall
            return x
        else:
            z = np.random.binomial(1, self.swap)
            if z == 1:
                return x
            else:
                return [nx,ny]

    def print_maze(self):
        maze = self.maze
        for i in range(self.M):
            for j in range(self.M):
                if maze[i,j] == 0:
                    print(" ")
                elif self.state is not None and self.state[0]==j and self.state[1]==i:
                    print("A")
                elif self.goal[0]==j and self.goal[1]==i:
                    print("G")
                elif maze[i,j] == 1:
                    print(".")
            print("")


def run_rl_acid_environment(env_name):

    if env_name == 'MAB':
        E = Environment()
        rewards = [0,0]
        counts = [0,0]
        for t in range(1000):
            x = E.start_episode()
            while x is not None:
                actions = E.get_actions()
                a = np.random.choice(actions)
                (x,r) = E.act(a)
                rewards[a] += r
                counts[a] += 1
        for a in [0,1]:
            assert (np.abs(np.float(rewards[a])/counts[a] -E.reward_dists[a]) < 0.1)

    if env_name == 'combolock':
        E = CombinationLock(horizon=3)
        print (E.opt)
        for t in range(10):
            x = E.start_episode()
            while x is not None:
                actions = E.get_actions()
                a = np.random.choice(actions)
                old = x
                (x,r) = E.act(a)
                print(old, a, r, x)

    if env_name == 'stochcombolock':
        E = StochasticCombinationLock(horizon=3, swap=0.5)
        print (E.opt_a)
        print (E.opt_b)
        for t in range(10):
            x = E.start_episode()
            while x is not None:
                actions = E.get_actions()
                a = np.random.choice(actions)
                old = x
                (x,r) = E.act(a)
                print(old, a, r, x)

    if env_name == 'maze':
        E = RandomGridWorld(M=3,swap=0.1, dim=2, noise=0.0)
        T = 0
        while True:
            T += 1
            x = E.start_episode()
            if T % 100 == 0:
                print("Iteration t = %d" % (T))
            while x is not None:
                E.print_maze()
                print(x)
                actions = E.get_actions()
                a = np.random.choice(len(actions))
                (x,r) = E.act(actions[a])
                if r == 1:
                    print("Success: T = %d" % (T))

