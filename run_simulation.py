from simulation import *
from environment import Environment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

env = Environment()
value_function_A = Mock_Value_Function(0,env)

agent_A = Agent(env, value_function_A, 0)
value_function_B = Mock_Value_Function(1,env)
agent_B = Agent(env, value_function_B, 1)
Simulation(env, agent_A, agent_B, True, True, False).run()

"""
env = Environment()
policy_A = LearnedMiniMaxPolicy(env, 0, 0.1)
value_function_A = MinimaxQ_Function(policy_A, start_value=1)
agent_A = Agent(env, value_function_A, 0)

policy_B = LearnedMiniMaxPolicy(env, 1, 0.1)
value_function_B = MinimaxQ_Function(policy_B, start_value=1)
agent_B = Agent(env, value_function_B, 1)
Simulation(env, agent_A, agent_B, True, True).run()"""