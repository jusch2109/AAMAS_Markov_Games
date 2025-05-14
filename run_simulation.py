from simulation import *
from environment import Environment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

env = Environment()
run_random = False
if run_random:
    agentA = Agent(env, RandomPolicy_Value_Function(0), 0)
    agentB = Agent(env, RandomPolicy_Value_Function(1), 1)
    Simulation(env, agentA, agentB).run()

else:
    agentA = Agent(env, Q_Function(EpsilonGreedyPolicy(),None), 0)
    agentB = Agent(env, Q_Function(EpsilonGreedyPolicy(),None), 1)
    Simulation(env, agentA, agentB).run()
