from simulation import *
from environment import Environment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

env = Environment()

mode = "test"  # "train" or "test" or "play" ... todo: finish this

if mode == "test":
    # test greedy vs random
    TRAINING = False
    USE_GUI = False
    IS_MAC = False
    policy_A = GreedyPolicy([],0)
    value_Function_A = Q_Function(policy_A)

    agent_A = Agent(env, value_Function_A, 0)

    policy_B = RandomPolicy(1)
    value_Function_B = RandomPolicy_Value_Function(1)
    agent_B = Agent(env, value_Function_B, 1)

    Simulation(env, agent_A, agent_B, TRAINING, USE_GUI,IS_MAC).run()

if mode == "train":
    TRAINING = True
    USE_GUI = False
    IS_MAC = False
    #Training thing:
    policy_A = EpsilonGreedyPolicy(0.3)
    value_Function_A = Q_Function(policy_A)
    agent_A = Agent(env, value_Function_A, 0)
    policy_B = EpsilonGreedyPolicy(0.3)
    value_Function_B = Q_Function(policy_B)
    agent_B = Agent(env, value_Function_B, 1)
    Simulation(env, agent_A, agent_B, TRAINING, USE_GUI,IS_MAC).run()


"""
value_function_A = Mock_Value_Function(0,env)

agent_A = Agent(env, value_function_A, 0)
value_function_B = Mock_Value_Function(1,env)
agent_B = Agent(env, value_function_B, 1)
Simulation(env, agent_A, agent_B, TRAINING, USE_GUI,IS_MAC).run()
"""

"""
#play vs Q-learning
policy_A = EpsilonGreedyPolicy()
value_Function_A = Q_Function(policy_A)
agent_A = Agent(env, value_Function_A, 0)

value_function_B = Mock_Value_Function(1,env)
agent_B = Agent(env, value_function_B, 1)
Simulation(env, agent_A, agent_B, TRAINING, USE_GUI,IS_MAC).run()

"""



"""
env = Environment()
policy_A = LearnedMiniMaxPolicy(env, 0, 0.1)
value_function_A = MinimaxQ_Function(policy_A, start_value=1)
agent_A = Agent(env, value_function_A, 0)

policy_B = LearnedMiniMaxPolicy(env, 1, 0.1)
value_function_B = MinimaxQ_Function(policy_B, start_value=1)
agent_B = Agent(env, value_function_B, 1)
Simulation(env, agent_A, agent_B, TRAINING, USE_GUI,IS_MAC).run()"""