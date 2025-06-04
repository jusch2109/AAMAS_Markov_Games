from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function, Handcrafted_Value_Function, JAL_AM_Q_Function
from policy import EpsilonGreedyPolicy, QPolicy, MockPolicy, ProbabilisticQPolicy, LearnedMiniMaxPolicy, HandcraftedPolicy
from agent import Agent
from value_function import MinimaxQ_Function
import run_simulation 

"""
SOCCER
"""

def s_train_challengerJALMR():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxrm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxrm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_soccer_jalmr.json"))

def s_train_challengerJALMM():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxmm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_soccer_jalmm.json"))

def s_train_challengerJALQR():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqer.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_soccer_jalqr.json"))

def s_train_challengerJALQQ():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqee.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_soccer_jalqq.json"))

def s_test_challengerJALMR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_soccer_jalmr.json"))

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxrm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxrm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer JAL MR:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

def s_test_challengerJALMM():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_soccer_jalmm.json"))

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxmm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer JAL MM:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

def s_test_challengerJALQR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_soccer_jalqr.json"))

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqer.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer JAL QR:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

def s_test_challengerJALQQ():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_soccer_jalqq.json"))

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqee.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer JAL QQ:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

"""
CATCH
"""
def c_train_challengerJALMR():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxrm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxrm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_catch_jalmr.json"))

def c_train_challengerJALMM():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxmm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_catch_jalmm.json"))

def c_train_challengerJALQR():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqer.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_catch_jalqr.json"))

def c_train_challengerJALQQ():
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #JAL train against qr
    policy_A = JAL_AM_Policy(epsilon=0.1, decay=0.9999954) 
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0.2,discount_factor=0.9)

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqee.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

    value_Function_A.save_dict(os.path.join("jal_models",f"0_catch_jalqq.json"))

def c_test_challengerJALMR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_catch_jalmr.json"))

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxrm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxrm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("catch JAL MR:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

def c_test_challengerJALMM():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_catch_jalmm.json"))

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxmm.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("catch JAL MM:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

def c_test_challengerJALQR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_catch_jalqr.json"))

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqer.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("catch JAL QR:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

def c_test_challengerJALQQ():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    policy_A = JAL_AM_Policy(epsilon=0, decay=1)
    value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
    value_Function_A.load_dict(os.path.join("jal_models",f"0_catch_jalqq.json"))

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqee.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("catch JAL QQ:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)


def main():
    s_train_challengerJALMR()
    s_train_challengerJALMM()
    s_train_challengerJALQR()
    s_train_challengerJALQQ()

    s_test_challengerJALMR()
    s_test_challengerJALMM()
    s_test_challengerJALQR()
    s_test_challengerJALQQ()

    c_train_challengerJALMR()
    c_train_challengerJALMM()
    c_train_challengerJALQR()
    c_train_challengerJALQQ()

    c_test_challengerJALMR()
    c_test_challengerJALMM()
    c_test_challengerJALQR()
    c_test_challengerJALQQ()


# Using the special variable 
# __name__
if __name__=="__main__":
    main()