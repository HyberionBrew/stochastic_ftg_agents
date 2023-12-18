from agents.agents_numpy import StochasticContinousFTGAgent
import torch
class StochasticContinousFTGTorchWrapper(StochasticContinousFTGAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(self, obs, action=None, std=None):
        # convert obs from torch to numpy, obs is a dictionary containing torch arrays
        print(obs)
        # loop over obs keys and transform to numpy from torch
        np_obs = {}
        print(obs)
        for key in obs.keys():
            np_obs[key] = obs[key].detach().numpy()
        
        _, action, log_probs = super().__call__(np_obs, action=action, std=None)
        # convert action from numpy to torch
        action = torch.from_numpy(action)
        # convert log_probs from numpy to torch
        log_probs = torch.from_numpy(log_probs)
        return None, action, log_probs