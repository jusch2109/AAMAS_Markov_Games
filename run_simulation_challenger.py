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

def s_train_challengerQR():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #q trained against qr
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqer.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)
     
    policy_A.save_dict(os.path.join("models",f"0_soccer_pi_qrx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_soccer_qrx.json"))


def s_train_challengerQQ():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #q trained against qq 
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B =  EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqee.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)
    
    policy_A.save_dict(os.path.join("models",f"0_soccer_pi_qqx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_soccer_qqx.json"))
    
def s_train_challengerMR():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #q trained against mr
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxrm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxrm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)
 
    policy_A.save_dict(os.path.join("models",f"0_soccer_pi_mrx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_soccer_mrx.json"))

def s_train_challengerMM():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()

    #q trained against mm 
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxmm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    r = simulation.run(timesteps)

    policy_A.save_dict(os.path.join("models",f"0_soccer_pi_mmx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_soccer_mmx.json"))

def s_test_challengerQR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    #qr tested against qr challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_soccer_qrx.json")) 

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqer.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer QR:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

def s_test_challengerQQ():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    #qq tested against qq challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_soccer_qqx.json")) 

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_soccer_gqee.json"))  

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer QQ:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)


def s_test_challengerMR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    #mr tested against mr challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_soccer_mrx.json")) 

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxrm.json"))
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxrm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer MR:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

def s_test_challengerMM():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = SoccerEnvironment()

    #mm/mr tested against mm/mr challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_soccer_mmx.json")) 

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_soccer_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_soccer_minimaxmm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Soccer MM:")
    simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    r = simulation.run(timesteps)

"""
CATCH
"""

def c_train_challengerQR():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #q trained against qr
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqer.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)
     
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps) 
    policy_A.save_dict(os.path.join("models",f"0_catch_pi_qrx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_catch_qrx.json"))


def c_train_challengerQQ():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #q trained against qq 
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqee.json"))

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)
    
    policy_A.save_dict(os.path.join("models",f"0_catch_pi_qqx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_catch_qqx.json"))


def c_train_challengerMR():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #q trained against mr
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxrm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxrm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

    policy_A.save_dict(os.path.join("models",f"0_catch_pi_mrx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_catch_mrx.json"))

def c_train_challengerMM():
    learning_rate = 0.1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    env = CatchEnvironment()

    #q trained against mm 
    policy_A = EpsilonGreedyPolicy(0.1)
    value_Function_A = Q_Function(policy_A, learning_rate=learning_rate, decay=decay)

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxmm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=True, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

    policy_A.save_dict(os.path.join("models",f"0_catch_pi_mmx.json"))
    value_Function_A.save_dict(os.path.join("models",f"0_catch_mmx.json"))

def c_test_challengerQR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    #qr tested against qr challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_catch_qrx.json")) 

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqer.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch QR:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

def c_test_challengerQQ():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    #qq tested against qq challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_catch_qqx.json")) 

    policy_B = EpsilonGreedyPolicy(0)
    value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"0_catch_gqee.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch QQ:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)


def c_test_challengerMR():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    #mr tested against mr challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_catch_mrx.json"))  

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxrm.json"))
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxrm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch MR:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)

def c_test_challengerMM():
    decay = 1
    explore_decay = decay
    timesteps = 100000
    env = CatchEnvironment()

    #mm/mr tested against mm/mr challenger
    policy_A = EpsilonGreedyPolicy(0)
    value_Function_A = Q_Function(policy_A, learning_rate=0, decay=1)
    value_Function_A.load_dict(os.path.join("models",f"0_catch_mmx.json"))  

    policy_B = LearnedMiniMaxPolicy(env, 1, 0)
    policy_B.load_dict(os.path.join("models",f"1_catch_pi_minimaxmm.json")) 
    value_Function_B = MinimaxQ_Function(policy_B, learning_rate=0, decay=1)
    value_Function_B.load_dict(os.path.join("models",f"1_catch_minimaxmm.json")) 

    agentA = Agent(env, value_Function_A, 0)
    agentB = Agent(env, value_Function_B, 1)

    print("Catch MM:")
    simulation_catch = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=False, use_gui=False, mac=False)
    rc = simulation_catch.run(timesteps)


def main():
    #s_train_challengerMR()
    #s_train_challengerMM()
    #s_train_challengerQR()
    #s_train_challengerQQ()

    s_test_challengerMR()
    s_test_challengerMM()
    s_test_challengerQR()
    s_test_challengerQQ()

    #c_train_challengerMR()
    #c_train_challengerMM()
    #c_train_challengerQR()
    #c_train_challengerQQ()

    c_test_challengerMR()
    c_test_challengerMM()
    c_test_challengerQR()
    c_test_challengerQQ()

# Using the special variable 
# __name__
if __name__=="__main__":
    main()