class EnvironmentConfigManipulator:
    """ Change the config based on the environment """

    @staticmethod
    def manipulate_config(config):

        # environment-specific arguments
        if config.env == 'mario':
            config.n_input_channels = 1
            config.n_actions = 5
            config.image_subsample = 4
            config.height = 60
            config.width = 64
            config.phi_layer_size = 16 * config.n_feature_maps
        elif config.env == 'montezuma':
            config.n_input_channels = 1
            config.n_actions = 18
            config.height = 80
            config.width = 80
            config.image_subsample = 2
            config.phi_layer_size = 25 * config.n_feature_maps
        else:
            raise NotImplementedError()
