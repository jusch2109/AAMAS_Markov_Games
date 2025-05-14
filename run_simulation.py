from simulation import Simulation
from environment import Environment
from policy import RandomPolicy
from value_function import Value_Function
from agent import Agent
env = Environment()
Simulation(env, Agent(RandomPolicy(env,0), Value_Function()), Agent(RandomPolicy(env,1),Value_Function())).run()