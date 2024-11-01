# import for dealing with .json
import json
from f110_agents.agents_numpy import StochasticContinousFTGAgent
from f110_agents.pure_pursuit import StochasticContinousPPAgent
from f110_agents.agents_numpy import DoubleAgentWrapper
import os
class Agent(object):
    def __init__(self):
        pass
    def load(self, config=None,name=None, no_print=False):
        # check the agent name and load the correct agent
        # load from our config file
        assert config or name is not None
        if config is None:
            # path of this file
            
            path = os.path.dirname(os.path.realpath(__file__))
            # go one up
            path = os.path.dirname(path)
            # go into agent_configs
            path = os.path.join(path, "agent_configs")
            # add the name
            config = os.path.join(path, name + ".json")
        with open(config, 'r') as config_file:
            data = json.load(config_file)

        agent_class = data.get('agent_class')
        if agent_class == "FTGAgent":
            parameters = data.get('agent_parameters')
            print("Agent parameters", parameters)
            if not no_print:
                print("Agent parameters", parameters)
            return StochasticContinousFTGAgent(**parameters)
        if agent_class == "PPAgent":
            parameters = data.get('agent_parameters')
            if not no_print:
                print("Agent parameters", parameters)
            return StochasticContinousPPAgent(**parameters)
        if agent_class == "SwitchingAgent":
            parameters = data.get('agent_parameters')
            # need to call load on the parameters agent1 and agent2
            parameters['agent1'] = self.load(parameters['agent1'])
            parameters['agent2'] = self.load(parameters['agent2'])
            return DoubleAgentWrapper(parameters['agent1'], parameters['agent2'], parameters['switching_timestep'])
        