from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function, Handcrafted_Value_Function, JAL_AM_Q_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy, QPolicy, LearnedMiniMaxPolicy, HandcraftedPolicy
from agent import Agent
from value_function import MinimaxQ_Function
import run_simulation 

def train_challengerJALQQ():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #JAL train against qq
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    #policy_B = RandomPolicy(1)
    #value_Function_B = RandomPolicy_Value_Function(1)

    policy_B = QPolicy({}, 1, explore)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_qqq.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)

    value_Function_A.save_dict(os.path.join("models",f"0_soccer_jalr.json"))

def test_challengerJALQQ():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("models",f"0_soccer_jalqqx.json"))

    #policy_B = QPolicy({}, 1, 0)
    #value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    #value_Function_B.load_dict(os.path.join("models",f"1_soccer_qqq.json"))

    policy_B = RandomPolicy(1)
    value_Function_B = RandomPolicy_Value_Function(1)

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer JAL QQ:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

def main():
    #train_challengerJALQQ()
    test_challengerJALQQ()

# Using the special variable 
# __name__
if __name__=="__main__":
    main()