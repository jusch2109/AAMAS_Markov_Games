from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function, JAL_AM_Q_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

# ToDo: Add QQ and QR

def training(policy,steps):
    """
    Helper function for training the agents depending on the policies
    """
    TRAINING = True
    USE_GUI = False
    IS_MAC = False
    match policy:
        case "MR":
            #minimax-Q trained against random
            env = SoccerEnvironment()

            policy_A = LearnedMiniMaxPolicy(env, 0, 0.2)
            value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=0.2, decay=0.9999954)
            agent_A = Agent(env, value_Function_A, 0)
            policy_B = RandomPolicy(1)
            value_Function_B = RandomPolicy_Value_Function(1)
            agent_B = Agent(env, value_Function_B, 1)

            SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999954, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(steps)

        case "MM":
            #minimax-Q trained against minimax-Q
            env = SoccerEnvironment()

            policy_A = LearnedMiniMaxPolicy(env, 0, 0.2)
            value_Function_A = MinimaxQ_Function(policy_A, start_value=1, learning_rate=0.2, decay=0.9999954)
            agent_A = Agent(env, value_Function_A, 0)

            policy_B = LearnedMiniMaxPolicy(env, 1, 0.2)
            value_Function_B = MinimaxQ_Function(policy_B, start_value=1, learning_rate=0.2, decay=0.9999954)
            agent_B = Agent(env, value_Function_B, 1)

            SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999954, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(steps)

        case "QR":
            #Q trained against random
            env = SoccerEnvironment()

            policy_A = QPolicy(q_table=defaultdict(lambda: defaultdict(float)),agent = 0)
            value_Function_A = Q_Function(policy_A, start_value=1)
            agent_A = Agent(env, value_Function_A, 0)

            policy_B = RandomPolicy(1)
            value_Function_B = RandomPolicy_Value_Function(1)
            agent_B = Agent(env, value_Function_B, 1)

            SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999954, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(steps)

        case "QQ":
            #Q trained against Q
            env = SoccerEnvironment()

            policy_A = QPolicy(q_table=defaultdict(lambda: defaultdict(float)),agent = 0, epsilon=0.2)
            value_Function_A = Q_Function(policy_A, start_value=1)
            agent_A = Agent(env, value_Function_A, 0)

            policy_B = QPolicy(q_table=defaultdict(lambda: defaultdict(float)),agent = 1, epsilon=0.2)
            value_Function_B = Q_Function(policy_B, start_value=1)
            agent_B = Agent(env, value_Function_B, 1)

            SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999954, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(steps)

        case "MR_challenger":
            env = SoccerEnvironment()

            # agent A follows the QR policy
            policy_A = QPolicy(q_table=defaultdict(lambda: defaultdict(float)),agent = 0, epsilon=0.2)
            value_Function_A = Q_Function(policy_A, start_value=1)
            agent_A = Agent(env, value_Function_A, 0)

            # agent B has fixed MR policy
            policy_B = LearnedMiniMaxPolicy(env,1, 0)
            policy_B.load_dict(f"soccer_pi_minimax_MR.json")
            value_Function_B = MinimaxQ_Function(policy_B, start_value=1)
            value_Function_B.load_dict(f"soccer_minimax_MR.json")
            agent_B = Agent(env, value_Function_B, 1)

            SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999954, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(steps)

        case "MM_challenger":
            env = SoccerEnvironment()

            # agent A follows the QR policy
            policy_A = QPolicy(q_table=defaultdict(lambda: defaultdict(float)),agent = 0)
            value_Function_A = Q_Function(policy_A, start_value=1)
            agent_A = Agent(env, value_Function_A, 0)

            # agent B has fixed MM policy
            policy_B = LearnedMiniMaxPolicy(env,1, 0)
            policy_B.load_dict(f"soccer_pi_minimax_MM.json")
            value_Function_B = MinimaxQ_Function(policy_B, start_value=1)
            value_Function_B.load_dict(f"soccer_minimax_MM.json")
            agent_B = Agent(env, value_Function_B, 1)

            SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999954, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(steps)

        case "JAL_AM_challenger":
            env = SoccerEnvironment()

            #Testtraining against Random

            policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
            value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.1,discount_factor=0.9)
            agent_A = Agent(env, value_Function_A, 0)

            policy_B = RandomPolicy(1)
            value_Function_B = RandomPolicy_Value_Function(1)
            agent_B = Agent(env, value_Function_B, 1)

            SoccerSimulation(env, agent_A, agent_B, explore_decay=0.9999954, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC).run(steps)

        # ToDo: training challenger

    if not policy == "JAL_AM_challenger":
        policy_A.save_dict(f"soccer_pi_{policy}.json")
    value_Function_A.save_dict(f"soccer_{policy}.json")

"""
Set Variables for training
"""
explore = 0.2
decy = 0.9999954
steps = 10000 #10**6

"""
Invoke training function for each polity
"""
#print("Training MR:")
#training("MR", steps)
#print("Training MM:")
#training("MM", steps)
#print("Training QR:")
#training("QR", steps)
#print("Training QQ:")
#training("QQ", steps)
#print("Training QQ:")
#training("QQ", steps)


"""
Invoke training function for each champion challenger
"""
print("Training JAL_AM_challenger:")
training("JAL_AM_challenger", steps)
#print("Training MR_challenger:")
#training("MR_challenger", steps)

