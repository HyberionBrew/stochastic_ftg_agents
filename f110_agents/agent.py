# import for dealing with .json
import json
from f110_agents.agents_numpy import StochasticContinousFTGAgent
class Agent(object):
    def __init__(self):
        pass
    def load(self, config):
        # check the agent name and load the correct agent
        with open(config, 'r') as config_file:
            data = json.load(config_file)
        agent_class = data.get('agent_class')
        if agent_class == "FTGAgent":
            parameters = data.get('agent_parameters')
            print("Agent parameters", parameters)
            return StochasticContinousFTGAgent(**parameters)
        