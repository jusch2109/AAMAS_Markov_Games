from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function, Handcrafted_Value_Function, JAL_AM_Q_Function
from policy import EpsilonGreedyPolicy, QPolicy, MockPolicy, ProbabilisticQPolicy, LearnedMiniMaxPolicy, HandcraftedPolicy, JAL_AM_Policy
from agent import Agent
from value_function import MinimaxQ_Function



def get_old_extra(extra):
    returner = []
    for extra_ele in extra:
        last = range(len(extra_ele))
        s_returner = ""
        for index in last:
            if extra_ele[index] == "p":
                s_returner += "q"
            elif extra_ele[index] == "q":
                s_returner += "g"
            else:
                s_returner += extra_ele[index]
        returner.append(s_returner)
    return returner

def get_old_type(type):
    if type == "probabilistic_q":
        return "q"
    elif type == "q":
        return "greedy"
    else:
        return type


def get_policy(type,id, env,explore,load_dicts=False, extra=""):
    """
    Returns a policy of the specified type and id.
    """
    if isinstance(env, CatchEnvironment):
        env_type = "catch"
    else:
        env_type = "soccer"

    if type == "random":
        print("Using RandomPolicy")
        p = RandomPolicy(id)
    elif type == "greedy":
        print("Using QPolicy")
        p = QPolicy([], id)
        if load_dicts:
            p.load_dict(os.path.join("models",f"{id}_{env_type}_pi_gq{extra}.json"))

    elif type == "epsilon_greedy":
        print("Using EpsilonGreedyPolicy")
        p = EpsilonGreedyPolicy(explore)
    elif type == "q":
        print("Using ProbabilisticQPolicy")
        p = ProbabilisticQPolicy({}, id, explore)
        if load_dicts:
            p.load_dict(os.path.join("models",f"{id}_{env_type}_pi_q{extra}.json"))
    elif type == "handcrafted":
        print("Using HandcraftedPolicy")
        is_soccer = True
        if env_type == "catch":
            is_soccer = False
        p = HandcraftedPolicy(id, is_soccer=is_soccer)
    elif type == "minimax":
        print("Using LearnedMiniMaxPolicy")
        p = LearnedMiniMaxPolicy(env, id, explore)
        if load_dicts:
            p.load_dict(os.path.join("models",f"{id}_{env_type}_pi_minimax{extra}.json"))
    elif type == "mock":
        print("Using MockPolicy")
        p = MockPolicy(id,env)
        if load_dicts:
            print("MockPolicy does not load dict.")
    else:
        raise ValueError(f"Unknown policy type: {type}")
    return p

def get_value_function(type, id, policy,learning_rate,decay,env, start_value = None, load_dicts=False,extra=""):
    if isinstance(env, CatchEnvironment):
        env_type = "catch"
    else:
        env_type = "soccer"

    if start_value is None:
        if isinstance(policy, LearnedMiniMaxPolicy):
            start_value = 1
        else:
            start_value = 0

    if type == "random":
        value_Function_B = RandomPolicy_Value_Function(id)
        if load_dicts:
            print("Random_Function does not load dict.")
    elif type == "handcrafted":
        is_soccer = True
        if env_type == "catch":
            is_soccer = False
        value_Function_B = Handcrafted_Value_Function(id, is_soccer=is_soccer)
        if load_dicts:
            print("Handcrafted_Value_Function does not load dict.")
    elif type == "mock":
        value_Function_B = Mock_Value_Function(id, policy)
        if load_dicts:
            print("Mock_Value_Function does not load dict.")
    elif type == "q" or type == "epsilon_greedy" or type == "greedy":
        value_Function_B = Q_Function(policy, start_value=start_value, learning_rate=learning_rate, decay=decay)
        if load_dicts:
            if type == "q":
                value_Function_B.load_dict(os.path.join("models",f"{id}_{env_type}_q{extra}.json"))
            else:
                ## if we saved it it was an epsilon greedy policy.

                last = range(len(extra))
                for index in last:
                    if extra[index] == "g":
                        extra = extra[:index] + "e" + extra[index + 1:]
                value_Function_B.load_dict(os.path.join("models",f"{id}_{env_type}_gq{extra}.json"))
    elif type == "minimax":
        value_Function_B = MinimaxQ_Function(policy, start_value=start_value, learning_rate=learning_rate, decay=decay)
        if load_dicts:
            value_Function_B.load_dict(os.path.join("models",f"{id}_{env_type}_minimax{extra}.json"))
            
    else:
        raise ValueError(f"Unknown value function type: {type}")
    return value_Function_B


def save_policies_and_value_functions(agentA, agentB, policyA, policyB, value_Function_A, value_Function_B, env, extra=["",""]):
    """
    Saves the policies and value functions of the agents.
    """
    if isinstance(env, CatchEnvironment):
        env_type = "catch"
    else:
        env_type = "soccer"
    if isinstance(policyA, LearnedMiniMaxPolicy):
        algoA = "minimax"
    elif isinstance(policyA, ProbabilisticQPolicy):
        algoA = "q"
    elif isinstance(policyA, EpsilonGreedyPolicy) or isinstance(policyA, QPolicy):
        ## make sure extra saves it as epsilon greedy in both cases
        last = range(len(extra))
        for index in last:
            if extra[index] == "g":
                extra = extra[:index] + "e" + extra[index + 1:]
        algoA = "gq"
    else:
        algoA = "random"
    if isinstance(policyB, LearnedMiniMaxPolicy):
        algoB = "minimax"
    elif isinstance(policyB, ProbabilisticQPolicy):
        algoB = "q"
    elif isinstance(policyB, EpsilonGreedyPolicy) or isinstance(policyB, QPolicy):
        ## make sure extra saves it as epsilon greedy in both cases
        last = range(len(extra))
        for index in last:
            if extra[index] == "g":
                extra = extra[:index] + "e" + extra[index + 1:]
        algoB = "gq"
    else:
        algoB = "random"
        
    policyA.save_dict(os.path.join("models",f"{agentA.agent_index}_{env_type}_pi_{algoA}{extra[0]}.json"))
    policyB.save_dict(os.path.join("models",f"{agentB.agent_index}_{env_type}_pi_{algoB}{extra[1]}.json"))
    
    value_Function_A.save_dict(os.path.join("models",f"{agentA.agent_index}_{env_type}_{algoA}{extra[0]}.json"))
    value_Function_B.save_dict(os.path.join("models",f"{agentB.agent_index}_{env_type}_{algoB}{extra[1]}.json"))
    return

def get_agents_policies_value_functions(type1, type2, env, explore, learning_rate, decay, load_dicts=False,extra=["",""]):
    """
    Returns an agent of the specified types.
    """
    policy1 = get_policy(type1, 0, env, explore, load_dicts,extra=extra[0])
    policy2 = get_policy(type2, 1, env, explore, load_dicts,extra=extra[1])

    value_Function1 = get_value_function(type1, 0, policy1, learning_rate, decay, env, load_dicts=load_dicts,extra=extra[0])
    value_Function2 = get_value_function(type2, 1, policy2, learning_rate, decay, env, load_dicts=load_dicts,extra=extra[1])

    agent1 = Agent(env, value_Function1, 0)
    agent2 = Agent(env, value_Function2, 1)
    return agent1, agent2, policy1, policy2, value_Function1, value_Function2




def train(_A_type, _B_type,env, explore_decay,explore, learning_rate, decay, timesteps = 1000000, training=True, use_gui=False, mac=False, extra=""):
    """
    Runs the simulation for a million timesteps
    """
    A_type = get_old_type(_A_type)
    B_type = get_old_type(_B_type)
    agentA, agentB, policy_A, policy_B, value_Function_A, value_Function_B = get_agents_policies_value_functions(A_type, B_type, env,explore, learning_rate, decay,load_dicts=False,extra=get_old_extra(extra))
    if isinstance(env, CatchEnvironment):
        simulation = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=training, use_gui=use_gui, mac=mac)
    else:
        simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=training, use_gui=use_gui, mac=mac)
    r = simulation.run(timesteps)
    save_policies_and_value_functions(agentA, agentB, policy_A, policy_B, value_Function_A, value_Function_B, env, extra=get_old_extra(extra))
    return r

def test(_A_type, _B_type, env,explore_decay,explore, learning_rate, decay,timesteps=100000, use_gui=False, mac=False,extra="", jalm_test=False):
    """
    Runs the simulation for 100k timesteps.  explore decay stayed to make the arguments the same, but its always 1.
    """
    A_type = get_old_type(_A_type)
    B_type = get_old_type(_B_type)
    agentA, agentB, policy_A, policy_B, value_Function_A, value_Function_B = get_agents_policies_value_functions(A_type, B_type, env,explore, learning_rate, decay, load_dicts=True,extra=get_old_extra(extra))
    if jalm_test:
        policy_A = JAL_AM_Policy(epsilon=0, decay=1)
        value_Function_A = JAL_AM_Q_Function(policy=policy_A,env=env,agent_idx=0,start_value=0.0,learning_rate=0,discount_factor=1)
        value_Function_A.load_dict(os.path.join("jal_models",f"0_catch_jalqr.json"))
        policy_B = EpsilonGreedyPolicy(0)
        value_Function_B = Q_Function(policy_B, learning_rate=0, decay=1)
        value_Function_B.load_dict(os.path.join("models",f"0_catch_gqer.json"))

    if isinstance(env, CatchEnvironment):
        simulation = CatchSimulation(env, agentA, agentB, explore_decay=1, training=False, use_gui=use_gui, mac=mac)
    else:
        simulation = SoccerSimulation(env, agentA, agentB, explore_decay=1, training=False, use_gui=use_gui, mac=mac)
    r = simulation.run(timesteps)
    return r


    

def test_trained():
    learning_rate = 1
    explore = 0
    decay = 1
    explore_decay = decay
    timesteps = 100000 
    
    soccer_results = {}
    catch_results = {}

    for A_type in ["random", "probabilistic_q", "q", "minimax", "handcrafted"]:
        for B_type in ["random", "probabilistic_q", "q", "minimax", "handcrafted"]:
            extra = [str(A_type[0]+B_type[0]), str(A_type[0]+B_type[0])]
            if A_type in ["random", "handcrafted"] and B_type not in ["random", "handcrafted"]:
                continue
            if A_type in ["probabilistic_q", "q", "minimax"] and B_type in ["probabilistic_q", "q", "minimax"]:
                extra = [str(A_type[0]+A_type[0]), str(B_type[0]+B_type[0])]    # use the ones trained against each other
            print(f"A: {A_type}, B: {B_type}")
            print("Soccer:")
            env = SoccerEnvironment()
            soccer_results[A_type + "_" + B_type] = test(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps,extra=extra)
            print("Catch:")
            env = CatchEnvironment()
            catch_results[A_type + "_" + B_type] = test(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps,extra=extra)

        print("Soccer Results:")
    print(soccer_results)
    print("Catch Results:")
    print(catch_results)


def train_test_specific(A_type, B_type, env):
    learning_rate = 1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000
    train(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps,extra=str(A_type[0]+B_type[0]))
    explore = 0
    decay = 1
    timesteps = 100000
    print("Catch:")
    env = CatchEnvironment()
    test(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps,extra=str(A_type[0]+B_type[0]), use_gui=False)

def show_agents():
    learning_rate = 1
    explore = 0
    decay = 1
    explore_decay = decay
    timesteps = 1000000
    env = SoccerEnvironment()
    env.state[2] = 1   # setting the random with the ball
    test("handcrafted", "random", env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=True, mac=False, extra=["hr","aa"])
    env.state[2] = 1   # setting the random with the ball
    ## QR vs R
    test("q", "random", env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=True, mac=False, extra=["qr","qr"])
    
    env = CatchEnvironment()
    env.state[2] = 0   # setting the handcrafted as the catcher
    test("handcrafted", "random", env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=True, mac=False, extra=["hr","aa"])
    env.state[2] = 0   # setting the minimax as the catcher
    ## MM VS MM
    test("minimax", "minimax", env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=True, mac=False, extra=["mm","mm"])
    env = CatchEnvironment()
    env.state[2] = 0   # setting jal as the catcher
    ## JAL AM vs Q
    ## this test is hardcoded, its not random random.
    test("random", "random", env, explore_decay, explore, learning_rate, decay, timesteps=timesteps,extra=[str("rr"), "rr"], use_gui=True, jalm_test=True)



def main():   
    learning_rate = 1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000   
    #env = SoccerEnvironment()
    #train("epsilon_greedy", "random", env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=False, mac=False, extra=["qr","qr"])

    show_agents()

def _main():
    types = ["random", "probabilistic_q", "minimax", "q", "mock", "handcrafted"]
    learning_rate = 1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    timesteps = 1000000    

    A_type = "minimax"
    B_type = "minimax"
    env = SoccerEnvironment()
    #train(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps,extra="")
    env = CatchEnvironment()
    #explore = 0
    print("handcrafted vs random catch")
    A_type = "handcrafted"
    B_type = "random"
    timesteps = 100000    
    test(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=False, mac=False, extra="")
    env = SoccerEnvironment()
    print("handcrafted vs random soccer")
    A_type = "handcrafted"
    B_type = "random"
    timesteps = 100000    
    test(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=False, mac=False,extra="")


# Using the special variable 
# __name__
if __name__=="__main__":
    main()
