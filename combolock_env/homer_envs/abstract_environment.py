class AbstractEnvironment:

    def reset(self):
        """ Reset the agent to a start state position (which
        can be deterministically or stochastically selected """
        raise NotImplementedError()

    def step(self, action):
        """ Take the action and return the new state, reward,
        a flag telling if the agent has reached a stop state
        and some meta information. """
        raise NotImplementedError()