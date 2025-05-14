from simulation import Simulation

env = Environment()
policy_A = LearnedMiniMaxPolicy(env)
value_function_A = MinimaxQ_Function(policy_A)
agent_A = Agent(env, value_function_A, 0)

policy_B = LearnedMiniMaxPolicy(env)
value_function_B = MinimaxQ_Function(policy_B)
agent_B = Agent(env, value_function_B, 1)
Simulation(env, agent_A, agent_B).run()