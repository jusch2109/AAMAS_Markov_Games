from simulation import *
from environment import SoccerEnvironment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function,Q_Function, Mock_Value_Function, Handcrafted_Value_Function
from policy import EpsilonGreedyPolicy, GreedyPolicy, MockPolicy, QPolicy, LearnedMiniMaxPolicy, HandcraftedPolicy
from agent import Agent
from value_function import MinimaxQ_Function

def get_policy(type,id, env,explore,load_dicts=False):
    """
    Returns a policy of the specified type and id.
    """
    if isinstance(env, CatchEnvironment):
        env_type = "catch"
    else:
        env_type = "soccer"

    if type == "random":
        p = RandomPolicy(id)
    elif type == "greedy":
        p = GreedyPolicy([], id)
        if load_dicts:
            p.load_dict(f"{id}_{env_type}_pi_gq.json")
    elif type == "epsilon_greedy":
        p = EpsilonGreedyPolicy(explore)
        if load_dicts:
            print("EpsilonGreedyPolicy does not load dict.")
    elif type == "q":
        p = QPolicy({}, id, explore)
        if load_dicts:
            p.load_dict(f"{id}_{env_type}_pi_q.json")
    elif type == "handcrafted":
        is_soccer = True
        if env_type == "catch":
            is_soccer = False
        p = HandcraftedPolicy(id, is_soccer=is_soccer)
    elif type == "minimax":
        p = LearnedMiniMaxPolicy(env, id, explore)
        if load_dicts:
            p.load_dict(f"{id}_{env_type}_pi_minimax.json")
    elif type == "mock":
        p = MockPolicy(id,env)
        if load_dicts:
            print("MockPolicy does not load dict.")
    else:
        raise ValueError(f"Unknown policy type: {type}")
    return p

def get_value_function(type, id, policy,learning_rate,decay,env, start_value = None, load_dicts=False):
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
                value_Function_B.load_dict(f"{id}_{env_type}_q.json")
            else:
                value_Function_B.load_dict(f"{id}_{env_type}_gq.json")
    elif type == "minimax":
        value_Function_B = MinimaxQ_Function(policy, start_value=start_value, learning_rate=learning_rate, decay=decay)
        if load_dicts:
            value_Function_B.load_dict(f"{id}_{env_type}_minimax.json")
    else:
        raise ValueError(f"Unknown value function type: {type}")
    return value_Function_B


def save_policies_and_value_functions(agentA, agentB, policyA, policyB, value_Function_A, value_Function_B, env):
    """
    Saves the policies and value functions of the agents.
    """
    if isinstance(env, CatchEnvironment):
        env_type = "catch"
    else:
        env_type = "soccer"
    if isinstance(policyA, LearnedMiniMaxPolicy):
        algoA = "minimax"
    elif isinstance(policyA, QPolicy):
        algoA = "q"
    elif isinstance(policyA, EpsilonGreedyPolicy) or isinstance(policyA, GreedyPolicy):
        algoA = "gq"
    else:
        algoA = "random"
    if isinstance(policyB, LearnedMiniMaxPolicy):
        algoB = "minimax"
    elif isinstance(policyB, QPolicy):
        algoB = "q"
    elif isinstance(policyB, EpsilonGreedyPolicy) or isinstance(policyB, GreedyPolicy):
        algoB = "gq"
    else:
        algoB = "random"
        
    policyA.save_dict(f"{agentA.agent_index}_{env_type}_pi_{algoA}.json")
    policyB.save_dict(f"{agentB.agent_index}_{env_type}_pi_{algoB}.json")
    
    value_Function_A.save_dict(f"{agentA.agent_index}_{env_type}_{algoB}.json")
    value_Function_B.save_dict(f"{agentB.agent_index}_{env_type}_{algoB}.json")
    return

def get_agents_policies_value_functions(type1, type2, env, explore, learning_rate, decay, load_dicts=False):
    """
    Returns an agent of the specified types.
    """
    policy1 = get_policy(type1, 0, env, explore, load_dicts)
    policy2 = get_policy(type2, 1, env, explore, load_dicts)

    value_Function1 = get_value_function(type1, 0, policy1, learning_rate, decay, env, load_dicts=load_dicts)
    value_Function2 = get_value_function(type2, 1, policy2, learning_rate, decay, env, load_dicts=load_dicts)

    agent1 = Agent(env, value_Function1, 0)
    agent2 = Agent(env, value_Function2, 1)
    return agent1, agent2, policy1, policy2, value_Function1, value_Function2




def train(A_type, B_type,env, explore_decay,explore, learning_rate, decay, timesteps = 1000000, training=True, use_gui=False, mac=False):
    """
    Runs the simulation for a million timesteps
    """
    agentA, agentB, policy_A, policy_B, value_Function_A, value_Function_B = get_agents_policies_value_functions(A_type, B_type, env,explore, learning_rate, decay,load_dicts=False)
    if isinstance(env, CatchEnvironment):
        simulation = CatchSimulation(env, agentA, agentB, explore_decay=explore_decay, training=training, use_gui=use_gui, mac=mac)
    else:
        simulation = SoccerSimulation(env, agentA, agentB, explore_decay=explore_decay, training=training, use_gui=use_gui, mac=mac)
    simulation.run(timesteps)
    save_policies_and_value_functions(agentA, agentB, policy_A, policy_B, value_Function_A, value_Function_B, env)
    return

def test(A_type, B_type, env,explore_decay,explore, learning_rate, decay,timesteps=100000, use_gui=False, mac=False):
    """
    Runs the simulation for 100k timesteps.  explore decay stayed to make the arguments the same, but its always 1.
    """
    agentA, agentB, policy_A, policy_B, value_Function_A, value_Function_B = get_agents_policies_value_functions(A_type, B_type, env,explore, learning_rate, decay, load_dicts=True)
    if isinstance(env, CatchEnvironment):
        simulation = CatchSimulation(env, agentA, agentB, explore_decay=1, training=False, use_gui=use_gui, mac=mac)
    else:
        simulation = SoccerSimulation(env, agentA, agentB, explore_decay=1, training=False, use_gui=use_gui, mac=mac)
    simulation.run(timesteps)
    return



def main():
    types = ["random", "greedy", "epsilon_greedy", "q", "minimax", "mock", "handcrafted"]
    learning_rate = 1
    explore = 0.2
    decay = 0.9999954
    explore_decay = decay
    A_type = "minimax"
    B_type = "minimax"
    env = SoccerEnvironment()
    timesteps = 1000000    
    #train(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps)
    env = CatchEnvironment()
    explore = 0
    A_type = "mock"
    B_type = "handcrafted"
    timesteps = 100000    
    test(A_type, B_type, env, explore_decay, explore, learning_rate, decay, timesteps=timesteps, use_gui=True, mac=False)

# Using the special variable 
# __name__
if __name__=="__main__":
    main()
