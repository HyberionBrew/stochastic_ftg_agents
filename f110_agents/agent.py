# import for dealing with .json
import json
from f110_agents.agents_numpy import StochasticContinousFTGAgent
from f110_agents.pure_pursuit import StochasticContinousPPAgent
from f110_agents.agents_numpy import DoubleAgentWrapper
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
        if agent_class == "PPAgent":
            parameters = data.get('agent_parameters')
            print("Agent parameters", parameters)
            return StochasticContinousPPAgent(**parameters)

        if agent_class == "SwitchingAgent":
            parameters = data.get('agent_parameters')
            # need to call load on the parameters agent1 and agent2
            parameters['agent1'] = self.load(parameters['agent1'])
            parameters['agent2'] = self.load(parameters['agent2'])
            return DoubleAgentWrapper(parameters['agent1'], parameters['agent2'], parameters['switching_timestep'])