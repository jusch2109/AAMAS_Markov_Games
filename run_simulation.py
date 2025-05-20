from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

env = SoccerEnvironment()

mode = "test_catch"  # "train" or "test" or "play" ... todo: finish this

if mode == "train_catch":
    env = CatchEnvironment()
    # Training thing:
    TRAINING = True
    USE_GUI = False
    IS_MAC = False
    #Training thing:
    policy_A = LearnedMiniMaxPolicy(env, 0, 0.2)
    value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=0.2, decay=0.99999)
    agent_A = Agent(env, value_Function_A, 0)
    policy_B = RandomPolicy(1)
    value_Function_B = RandomPolicy_Value_Function(1)
    agent_B = Agent(env, value_Function_B, 1)

    CatchSimulation(env, agent_A, agent_B, training=True).run(100000)
    policy_A.save_dict("catch_pi_minimax.json")
    value_Function_A.save_dict("catch_minimax.json")

if mode == "test_catch":
    env = CatchEnvironment()
    # Training thing:
    TRAINING = False
    USE_GUI = True
    IS_MAC = True

    policy_A = LearnedMiniMaxPolicy(env, 0, 0.2)
    policy_A.load_dict("catch_pi_minimax.json")
    value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=0.2, decay=0.99999)
    value_Function_A.load_dict("catch_minimax.json")
    agent_A = Agent(env, value_Function_A, 0)

    policy_B = RandomPolicy(1)
    value_Function_B = RandomPolicy_Value_Function(1)
    agent_B = Agent(env, value_Function_B, 1)

    CatchSimulation(env, agent_A, agent_B, training=True, use_gui=USE_GUI, mac=IS_MAC).run(1000000)

if mode == "test_soccer":
    # test greedy vs random
    TRAINING = False
    USE_GUI = True
    IS_MAC = True

    # load the trained policy

    policy_A = LearnedMiniMaxPolicy(env,0, 0)
    policy_A.load_dict("soccer_pi_minimax.json")
    value_Function_A = MinimaxQ_Function(policy_A, start_value=1)
    value_Function_A.load_dict("soccer_minimax.json")

    agent_A = Agent(env, value_Function_A, 0)

    policy_B = RandomPolicy(1)
    value_Function_B = RandomPolicy_Value_Function(1)
    agent_B = Agent(env, value_Function_B, 1)

    SoccerSimulation(env, agent_A, agent_B, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run()

if mode == "train_soccer":
    TRAINING = True
    USE_GUI = False
    IS_MAC = False
    #Training thing:
    policy_A = LearnedMiniMaxPolicy(env, 0, 0.2)
    value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=0.2, decay=0.99999)
    agent_A = Agent(env, value_Function_A, 0)
    policy_B = RandomPolicy(1)
    value_Function_B = RandomPolicy_Value_Function(1)
    agent_B = Agent(env, value_Function_B, 1)

    SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(100000)

    #policy_A.save_dict("soccer_pi_minimax.json")
    #value_Function_A.save_dict("soccer_minimax.json")


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