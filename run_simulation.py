from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function


min_max = False   # min max or q?
do_all = True

mode = "train_catch"  # "train" or "test" or "play" ... todo: finish this



learning_rate = 1
explore = 0.2
decay = 0.9999954


explore_decay = decay


for min_max in [False, True]:
    _save_name = "minimax" if min_max else "q"
    print("---------------------------------------------------\nstarting algo:")
    print(_save_name)
    if mode == "train_catch" or do_all:
        print("training catch")
        timesteps = 1000000
        env = CatchEnvironment()
        # Training thing:
        TRAINING = True
        USE_GUI = False
        IS_MAC = False
        #Training thing:
        if min_max:
            policy_A = LearnedMiniMaxPolicy(env, 0, explore)
            value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=learning_rate, decay=decay)
        else:
            policy_A = EpsilonGreedyPolicy(explore)
            value_Function_A = Q_Function(policy_A, start_value=0, learning_rate=learning_rate, decay=decay)
            
        agent_A = Agent(env, value_Function_A, 0)
        policy_B = RandomPolicy(1)
        value_Function_B = RandomPolicy_Value_Function(1)
        agent_B = Agent(env, value_Function_B, 1)

        CatchSimulation(env, agent_A, agent_B, explore_decay=explore_decay, training=TRAINING).run(timesteps)
        policy_A.save_dict(f"catch_pi_{_save_name}.json")
        value_Function_A.save_dict(f"catch_{_save_name}.json")

    if mode == "test_catch" or do_all:
        print("testing catch")
        timesteps = 100000
        env = CatchEnvironment()
        # Training thing:
        TRAINING = False
        USE_GUI = False
        IS_MAC = False


        if min_max:
            policy_A = LearnedMiniMaxPolicy(env, 0, 0) ## explore should be 0 right?
            policy_A.load_dict(f"catch_pi_{_save_name}.json")
            value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=learning_rate, decay=decay)
            value_Function_A.load_dict(f"catch_{_save_name}.json")
        else:
            policy_A = GreedyPolicy([],0)
            policy_A.load_dict(f"catch_pi_{_save_name}.json")
            value_Function_A = Q_Function(policy_A, start_value=0, learning_rate=learning_rate, decay=decay)
            value_Function_A.load_dict(f"catch_{_save_name}.json")
            
        agent_A = Agent(env, value_Function_A, 0)

        policy_B = RandomPolicy(1)
        value_Function_B = RandomPolicy_Value_Function(1)
        agent_B = Agent(env, value_Function_B, 1)

        CatchSimulation(env, agent_A, agent_B, explore_decay=explore_decay, training=True, use_gui=USE_GUI, mac=IS_MAC).run(timesteps)

    if mode == "train_soccer" or do_all:
        print("training soccer")
        timesteps = 1000000
        env = SoccerEnvironment()
        TRAINING = True
        USE_GUI = False
        IS_MAC = False
        #Training thing:

        if min_max:
            policy_A = LearnedMiniMaxPolicy(env, 0, explore=explore)
            value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=learning_rate, decay=decay)
            policy_B = RandomPolicy(1)
            value_Function_B = RandomPolicy_Value_Function(1)
        else:
            #policy_A.load_dict(f"soccer_pi_{_save_name}.json")        
            policy_A = EpsilonGreedyPolicy(explore)
            value_Function_A = Q_Function(policy_A, start_value=0, learning_rate=learning_rate, decay=decay)
            
            policy_B = EpsilonGreedyPolicy(explore)
            value_Function_B = Q_Function(policy_B, start_value=0, learning_rate=learning_rate, decay=decay)

            
        agent_A = Agent(env, value_Function_A, 0)
        agent_B = Agent(env, value_Function_B, 1)
        # train for a milion timesteps
        SoccerSimulation(env, agent_A, agent_B, explore_decay=explore_decay, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(1000000)

        policy_A.save_dict(f"soccer_pi_{_save_name}.json")
        value_Function_A.save_dict(f"soccer_{_save_name}.json")



    if mode == "test_soccer" or do_all:
        print("testing soccer")
        timesteps = 100000
        env = SoccerEnvironment()
        # test greedy vs random
        TRAINING = False
        USE_GUI = False
        IS_MAC = False

        # load the trained policy

        if min_max:
            policy_A = LearnedMiniMaxPolicy(env,0, 0)
            policy_A.load_dict(f"soccer_pi_{_save_name}.json")
            value_Function_A = MinimaxQ_Function(policy_A, start_value=1)
            value_Function_A.load_dict(f"soccer_{_save_name}.json")
        else:
            policy_A = GreedyPolicy([],0)
            policy_A.load_dict(f"soccer_pi_{_save_name}.json")
            value_Function_A = Q_Function(policy_A, start_value=0, learning_rate=learning_rate, decay=decay)
            value_Function_A.load_dict(f"soccer_{_save_name}.json")

        agent_A = Agent(env, value_Function_A, 0)

        policy_B = RandomPolicy(1)
        value_Function_B = RandomPolicy_Value_Function(1)
        agent_B = Agent(env, value_Function_B, 1)

        SoccerSimulation(env, agent_A, agent_B, explore_decay=explore_decay, training=True, use_gui=USE_GUI, mac=IS_MAC).run(timesteps)
