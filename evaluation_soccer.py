from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function, JAL_AM_Q_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy
from agent import Agent
from policy import LearnedMiniMaxPolicy
from value_function import MinimaxQ_Function

def evaluation(policy, challenger,steps):

    """
    Function for running the evaluations depending on the policy of the agent and the challenger
    """

    TRAINING = False
    USE_GUI = False
    IS_MAC = False

    env = SoccerEnvironment()

    print(f"Agent A: {policy} vs. Agent B: {challenger}")

    match challenger:
        case "random":
            match policy:
                case "MM" | "MR":
                    policy_A = LearnedMiniMaxPolicy(env,0, 0)
                    policy_A.load_dict(f"soccer_pi_minimax_{policy}.json")
                    value_Function_A = MinimaxQ_Function(policy_A, start_value=1)
                    value_Function_A.load_dict(f"soccer_minimax_{policy}.json")
                    agent_A = Agent(env, value_Function_A, 0)
                
                case "QQ" | "QR":
                    policy_A = QPolicy({},agent = 0, epsilon=0.2)
                    value_Function_A = Q_Function(policy_A, start_value=1)
                    value_Function_A.load_dict(f"soccer_{policy}")
                    agent_A = Agent(env, value_Function_A, 0)

            policy_B = RandomPolicy(1)
            value_Function_B = RandomPolicy_Value_Function(1)
            agent_B = Agent(env, value_Function_B, 1)

    
        case "handbuild":

            print(f"Agent A: {policy} vs. Agent B: {challenger}")

            # similar to random 

        case "MR_challenger" | "MM_challenger":

            policy_A = LearnedMiniMaxPolicy(env,0, 0)
            policy_A.load_dict(f"soccer_pi_minimax_{policy}.json")
            value_Function_A = MinimaxQ_Function(policy_A, start_value=1)
            value_Function_A.load_dict(f"soccer_minimax_{policy}.json")
            agent_A = Agent(env, value_Function_A, 0)

            policy_B = LearnedMiniMaxPolicy(env,1, 0)
            policy_B.load_dict(f"soccer_pi_minimax_{challenger}.json")
            value_Function_B = MinimaxQ_Function(policy_A, start_value=1)
            value_Function_B.load_dict(f"soccer_minimax_{challenger}.json")
            agent_B = Agent(env, value_Function_B, 1)

        case "QR_challenger" | "QQ_challenger":

            print(f"Agent A: {policy} vs. Agent B: {challenger}")

            policy_A = QPolicy({},agent = 0)
            value_Function_A = Q_Function(policy_A, start_value=1)
            value_Function_A.load_dict(f"soccer_{policy}")
            agent_A = Agent(env, value_Function_A, 0)

            policy_B = QPolicy({},agent = 1)
            value_Function_B = Q_Function(policy_B, start_value=1)
            value_Function_B.load_dict(f"soccer_{challenger}")
            agent_B = Agent(env, value_Function_B, 1)

        case "JAL_AM_challenger":

            match policy:
                case "MM" | "MR":
                    policy_A = LearnedMiniMaxPolicy(env,0, 0)
                    policy_A.load_dict(f"soccer_pi_minimax_{policy}.json")
                    value_Function_A = MinimaxQ_Function(policy_A, start_value=1)
                    value_Function_A.load_dict(f"soccer_minimax_{policy}.json")
                    agent_A = Agent(env, value_Function_A, 0)
                
                case "QQ" | "QR":
                    policy_A = LearnedMiniMaxPolicy(env,0, 0.2) # Q learning
                    policy_A.load_dict(f"soccer_pi_minimax_{policy}.json") # Q learning
                    value_Function_A = Q_Function(policy_A, start_value=1) # Q learning
                    value_Function_A.load_dict(f"soccer_minimax_{policy}.json") # Q learning
                    agent_A = Agent(env, value_Function_A, 0)
            
            policy_B = JAL_AM_Policy(env,1, 0)
            policy_B.load_dict(f"soccer_pi_minimax_JAL_AM_challenger.json")
            value_Function_B = JAL_AM_Q_Function(policy_A, start_value=1)
            value_Function_B.load_dict(f"soccer_minimax_JAL_AM_challenger.json")
            agent_B = Agent(env, value_Function_B, 1)

    simulation = SoccerSimulation(env, agent_A, agent_B, explore_decay=1, training=TRAINING, use_gui=USE_GUI, mac=IS_MAC)
    simulation.run(steps)


"""
Invoking the evaluation function
"""
#print("Challenger: Random")
#evaluation_random = []
#evaluation_random.append(evaluation("MR","random",100000))
#evaluation_random.append(evaluation("MM","random",100000))
#evaluation_random.append(evaluation("QR","random",100000))
#evaluation_random.append(evaluation("QQ","random",100000))
#print(evaluation_random)

#print("Challenger: Handbuild")
#evaluation_handbuild = []
#evaluation_handbuild.append(evaluation("MR","handbuild",100000))
#evaluation_handbuild.append(evaluation("MM","handbuild",100000))
#evaluation_handbuild.append(evaluation("QR","handbuild",100000))
#evaluation_handbuild.append(evaluation("QQ","handbuild",100000))
#print(evaluation_random)

#print("Challenger: MR_challenger")
#evaluation_mr = []
#evaluation_mr.append(evaluation("MR","MR_challenger",100000))
#print(evaluation_mr)

#print("Challenger: MM_challenger")
#evaluation_mm = []
#evaluation_mm.append(evaluation("MM","MM_challenger",100000))
#print(evaluation_random)

print("Challenger: QR_challenger")
evaluation_qr = []
evaluation_qr.append(evaluation("QR","QR_challenger",100000))
print(evaluation_qr)

#print("Challenger: QQ_challenger")
#evaluation_qq = []
#evaluation_qq.append(evaluation("QQ","QQ_challenger",100000))
#print(evaluation_qq)

#print("Challenger: JAL_AM_challenger)
#evaluation_jam = []
#evaluation_jam.append(evaluation("MR","JAL_AM_challenger",100000))
#evaluation_jam.append(evaluation("MM","JAL_AM_challenger",100000))
#print(evaluation_jam)