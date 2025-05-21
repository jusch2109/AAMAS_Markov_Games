from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

explore = 0.2
decy = 0.9999954
steps = 10**6
TRAINING = True
USE_GUI = False
IS_MAC = False


env = CatchEnvironment()

policy_A = LearnedMiniMaxPolicy(env, 0, 0.2)
value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=0.2, decay=0.99999)
agent_A = Agent(env, value_Function_A, 0)
policy_B = RandomPolicy(1)
value_Function_B = RandomPolicy_Value_Function(1)
agent_B = Agent(env, value_Function_B, 1)

CatchSimulation(env, agent_A, agent_B, training=True).run(100000)
policy_A.save_dict("catch_pi_minimax.json")
value_Function_A.save_dict("catch_minimax.json")
