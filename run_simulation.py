from simulation import Simulation
from environment import Environment
from policy import RandomPolicy
from value_function import Value_Function, RandomPolicy_Value_Function
from agent import Agent
env = Environment()
Simulation(env, Agent(env, RandomPolicy_Value_Function(0),0), Agent(env,RandomPolicy_Value_Function(1), 1)).run()