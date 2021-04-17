
def default_config():
    return {'num_actions': 10,
            'actions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'horizon': 20,
            'optimal_reward': 5.0,
            'anti_shaping_reward': 0.0,
            'anti_shaping_reward2': 1.0,
            'trace_sample_rate': 500,
            'return_state': False,
            'save_trace': True,
            'noise': 'hadamhardg', 
            'obs_dim': -1,
            'trace_folder': './traces/',
            'feature_type': 'feature',
            'gamma': 1.0
            }


