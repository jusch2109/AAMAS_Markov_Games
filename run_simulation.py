from simulation import *
from environment import Environment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

env = Environment()
policy_A = LearnedMiniMaxPolicy(env, 0, 0.1)
value_function_A = MinimaxQ_Function(policy_A)
agent_A = Agent(env, value_function_A, 0)

policy_B = LearnedMiniMaxPolicy(env, 1, 0.1)
value_function_B = MinimaxQ_Function(policy_B)
agent_B = Agent(env, value_function_B, 1)
Simulation(env, agent_A, agent_B, True, True).run()