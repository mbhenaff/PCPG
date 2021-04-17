import gym

from collections import deque
#from skimage.color import rgb2gray
#from skimage.transform import resize
from homer_envs.abstract_environment import AbstractEnvironment
#from homer_envs.grid_world import GridWorld
from homer_envs.rl_acid_environment import *
#from homer_envs.visual_combolock import VisualComboLock
from homer_envs.noise_gen import get_sylvester_hadamhard_matrix_dim


class GenerateEnvironmentWrapper(AbstractEnvironment):
    """" Wrapper class for generating environments using names and config """

    OpenAIGym, RL_ACID, GRIDWORLD, VISUALCOMBOLOCK = range(4)

    def __init__(self, env_name, config, bootstrap_env=None):
        """
        :param env_name: Name of the environment to create
        :param config:  Configuration to use
        :param bootstrap_env: Environment used for defining
        """

        self.tolerance = 0.5
        self.env_type = None
        self.env_name = env_name
        self.config = config

        # A boolean flag indicating if we should save traces or not
        # A trace is a sequence of {(obs, state, action, reward, obs, state, ...., obs, state)}
        self.save_trace = config["save_trace"]
        self.trace_sample_rate = config["trace_sample_rate"]            # How many often should we save the traces
        self.trace_folder = config["trace_folder"]                      # Folder for saving traces
        self.trace_data = []                                            # Set of currently unsaved traces
        self.current_trace = None                                       # Current trace
        self.num_eps = 0                                                # Number of episodes passed
        self.sum_total_reward = 0.0                                     # Total reward received by the agent
        self._sum_this_episode = 0.0                                    # Total reward received in current episode
        self.moving_average_reward = deque(maxlen=100)                  # For moving average calculation
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(config["obs_dim"],),dtype=np.float)
        self.action_space = gym.spaces.Discrete(config["num_actions"])

        # Create a folder for saving traces
#        if not os.path.exists(self.trace_folder):
#            os.makedirs(self.trace_folder)

        if env_name == 'diabcombolock':
            # Diabolical Stochastic Combination Lock

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True
            self.reward_range = (0.0, config['optimal_reward'])
            self.metadata = None

            
            if config["noise"] == "none":

                self.noise_type = Environment.NONE
                assert config["obs_dim"] == 2 * config["horizon"] + 4
                
            elif config["noise"] == "bernoulli":

                self.noise_type = Environment.BERNOULLI
                assert config["obs_dim"] == 2 * config["horizon"] + 4, "Set obs_dim to -1 in config for auto selection"

            elif config["noise"] == "gaussian":

                self.noise_type = Environment.GAUSSIAN
                assert config["obs_dim"] == config["horizon"] + 4, "Set obs_dim to -1 in config for auto selection"

            elif config["noise"] == "hadamhard":

                self.noise_type = Environment.HADAMHARD
                assert config["obs_dim"] == get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4), \
                    "Set obs_dim to -1 in config for auto selection"

            elif config["noise"] == "hadamhardg":

                self.noise_type = Environment.HADAMHARDG
                assert config["obs_dim"] == get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4), \
                    "Set obs_dim to -1 in config for auto selection"

            else:
                raise AssertionError("Unhandled noise type %r" % config["noise"])

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = DiabolicalCombinationLock(horizon=config["horizon"], swap=0.5,
                                                     num_actions=10, anti_shaping_reward=config['anti_shaping_reward'],
                                                     noise_type=self.noise_type, optimal_reward=config['optimal_reward'], anti_shaping_reward2=config['anti_shaping_reward2'], seed=config['seed'])

            # Reach the two states with probability at least 0.25 each and the third state with probability at least 0.5
            self.homing_policy_validation_fn = lambda dist, step: \
                str((0, step)) in dist and str((1, step)) in dist and str((2, step)) in dist and \
                dist[str((0, step))] + dist[str((1, step))] > 50 - self.tolerance and \
                dist[str((2, step))] > 50 - self.tolerance

        elif env_name == 'maze':
            # Maze world

            self.env_type = GenerateEnvironmentWrapper.RL_ACID
            self.thread_safe = True

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = RandomGridWorld(M=3, swap=0.1, dim=2, noise=0.0)

            self.homing_policy_validation_fn = None

        elif env_name == 'montezuma':
            # Montezuma Revenge

            self.env_type = GenerateEnvironmentWrapper.OpenAIGym
            self.thread_safe = True
            self.num_repeat_action = 4  # Repeat each action these many times.

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = gym.make('MontezumaRevengeDeterministic-v4')

            # Since we don't have access to underline state in this problem, we cannot define a validation function
            self.homing_policy_validation_fn = None

        elif env_name == 'gridworld' or env_name == 'gridworld-feat':
            # Grid World

            self.env_type = GenerateEnvironmentWrapper.GRIDWORLD
            self.thread_safe = True

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = GridWorld(num_grid_row=4, num_grid_col=4, horizon=config["horizon"], obs_dim=config["obs_dim"])

            reachable_states = self.env.get_reachable_states()
            num_states = self.env.get_num_states()

            self.homing_policy_validation_fn = lambda dist, step: all(
                [str(state) in dist and dist[str(state)] >= 1.0 / float(max(1, num_states)) - self.tolerance
                 for state in reachable_states[step]])

        elif env_name == 'visualcombolock':
            # Visual Combo Lock

            self.env_type = GenerateEnvironmentWrapper.VISUALCOMBOLOCK
            self.thread_safe = True

            if bootstrap_env is not None:
                self.env = bootstrap_env
            else:
                self.env = VisualComboLock(horizon=config["horizon"],
                                           swap=0.5,
                                           num_actions=10,
                                           anti_shaping_reward=0.1,
                                           obs_dim=config["obs_dim"],
                                           vary_instance=True)

            # TODO make this validation function stricter like for other combination lock environments
            self.homing_policy_validation_fn = lambda dist, step: \
                str((0, step)) in dist and str((1, step)) in dist and str((2, step)) in dist and \
                dist[str((0, step))] > 20 - self.tolerance and \
                dist[str((1, step))] > 20 - self.tolerance and \
                dist[str((2, step))] > 20 - self.tolerance

        else:
            raise AssertionError("Environment name %r not in RL Acid Environments " % env_name)

    def generate_homing_policy_validation_fn(self):

        if self.homing_policy_validation_fn is not None:
            return self.homing_policy_validation_fn

    def step(self, action):

        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:

            observation, reward, info = self.env.act(action)
            done = self.env.h == self.config['horizon']
            # TODO: check this is ok with Homer code
#            done = observation is None
            # TODO return non-none observation when observation is None.

            if self.save_trace:
                self.current_trace.extend([action, reward, observation, info["state"]])
            self._sum_this_episode += reward

            return observation, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.OpenAIGym:

            # Repeat the action K steps
            for _ in range(self.num_repeat_action):
                image, reward, done, info = self.env.step(action)
            image = self.openai_gym_process_image(image)
            assert "state" not in info
            info["state"] = self.openai_ram_for_state()
            return image, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.GRIDWORLD:

            image, reward, done, info = self.env.step(action)
            return image, float(reward), done, info

        elif self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:

            image, reward, done, info = self.env.step(action)
            return image, float(reward), done, info

        else:
            raise AssertionError("Unhandled environment type %r" % self.env_type)

    def reset(self):

        if self.env_type == GenerateEnvironmentWrapper.RL_ACID:

            self.sum_total_reward += self._sum_this_episode
            if self.num_eps > 0:
                self.moving_average_reward.append(self._sum_this_episode)
            self._sum_this_episode = 0.0

            if self.num_eps % 500 == 0:
                mean_result = self.sum_total_reward / float(max(1, self.num_eps))
                mean_moving_average = sum(self.moving_average_reward) / float(max(1, len(self.moving_average_reward)))
                with open(self.trace_folder + "/progress.csv", "a") as g:
                    if self.num_eps == 0:
                        g.write("Episodes Completed,   Mean Total Reward,      Mean Moving Average\n")
                    g.write("%d \t %f \t %f \n" % (self.num_eps, mean_result, mean_moving_average))

            self.num_eps += 1      # Current episode ID

            obs, info = self.env.start_episode()

            if self.save_trace:

                # Add current trace to list of traces at a certain rate
                if self.current_trace is not None and self.num_eps % self.trace_sample_rate == 0:
                    self.trace_data.append(self.current_trace)

                # Save data if needed
                if len(self.trace_data) == 1000:

                    with open(self.trace_folder + '/%s_%d' % (self.env_name, time.time()), 'wb') as f:       # TODO
                        pickle.dump(self.trace_data, f)
                    self.trace_data = []

                self.current_trace = [obs, info["state"]]
            if self.config['return_state']:
                return obs, info
            else:
                return obs

        elif self.env_type == GenerateEnvironmentWrapper.OpenAIGym:

            image = self.env.reset()
            image = self.openai_gym_process_image(image)
            return image, {"state": self.openai_ram_for_state()}

        elif self.env_type == GenerateEnvironmentWrapper.GRIDWORLD:

            return self.env.reset()

        elif self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:

            return self.env.reset()

        else:
            raise AssertionError("Unhandled environment type %r" % self.env_type)

    def openai_gym_process_image(self, image):

        if self.env_name == "montezuma":
            image = image[34: 34 + 160, :160]       # 160 x 160 x 3
            image = image/256.0

            if self.config["obs_dim"] == [1, 160, 160, 3]:
                return image
            elif self.config["obs_dim"] == [1, 84, 84, 1]:
                image = resize(rgb2gray(image), (84, 84), mode='constant')
                image = np.expand_dims(image, 2)  # 84 x 84 x 1
                return image
            else:
                raise AssertionError("Unhandled configuration %r" % self.config["obs_dim"])
        else:
            raise AssertionError("Unhandled OpenAI Gym environment %r" % self.env_name)

    def openai_ram_for_state(self):
        """ Create State for OpenAI gym using RAM. This is useful for debugging. """

        ram = self.env.env._get_ram()

        if self.env_name == "montezuma":
            # x, y position and orientation of agent, x-position of the skull and position of items like key
            state = "(%d, %d, %d, %d, %d)" % (ram[42], ram[43], ram[52], ram[47], ram[67])
            return state
        else:
            raise NotImplementedError()

    def get_optimal_value(self):

        if self.env_name == 'combolock' or self.env_name == 'stochcombolock' or \
                self.env_name == 'diabcombolock' or self.env_name == 'visualcombolock':
            return self.env.get_optimal_value()
        else:
            return None

    def is_thread_safe(self):
        return self.thread_safe

    @staticmethod
    def adapt_config_to_domain(env_name, config):
        """ This function adapts the config to the environment.
        """

        if config["obs_dim"] == -1:

            if env_name == 'combolock':
                config["obs_dim"] = 3 * config["horizon"] + 2

            elif env_name == 'stochcombolock' or env_name == 'diabcombolock':

                if config["noise"] == "bernoulli" or config["noise"] == 'none':
                    config["obs_dim"] = 2 * config["horizon"] + 4
                elif config["noise"] == "gaussian":
                    config["obs_dim"] = config["horizon"] + 4
                elif config["noise"] == "hadamhard":
                    config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)
                elif config["noise"] == "hadamhardg":
                    config["obs_dim"] = get_sylvester_hadamhard_matrix_dim(config["horizon"] + 4)
                else:
                    raise AssertionError("Unhandled noise type %r" % config["noise"])

            else:
                raise AssertionError("Cannot adapt to unhandled environment %s" % env_name)

    def get_bootstrap_env(self):
        """ Environments which are thread safe can be bootstrapped. There are two ways to do so:
        1. Environment with internal state which can be replicated directly.
            In this case we return the internal environment.
        2. Environments without internal state which can be created exactly from their name.
            In this case we return None """

        assert self.thread_safe, "To bootstrap it must be thread safe"
        if self.env_name == 'stochcombolock' or self.env_name == 'combolock' or \
                self.env_name == 'diabcombolock' or self.env_name == 'visualcombolock':
            return self.env
        else:
            return None

    def save_environment(self, folder_name, trial_name):

        if self.env_type == GenerateEnvironmentWrapper.RL_ACID or \
                self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:
            return self.env.save(folder_name + "/trial_%r_env" % trial_name)
        else:
            pass        # Nothing to save

    def load_environment_from_folder(self, env_folder_name):

        if self.env_type == GenerateEnvironmentWrapper.RL_ACID or \
                self.env_type == GenerateEnvironmentWrapper.VISUALCOMBOLOCK:
            self.env = self.env.load(env_folder_name)
        else:
            raise AssertionError("Cannot load environment for Non RL Acid settings")

    def is_deterministic(self):
        raise NotImplementedError()

    @staticmethod
    def make_env(env_name, config, bootstrap_env):
        return GenerateEnvironmentWrapper(env_name, config, bootstrap_env)
