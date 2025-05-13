import numpy as np
from environment import Environment
import random

class Policy():

    def getAction(self, state, *args, **kwargs):
        """
        Returns an action according to the policy based on the current state
        """
        #abstract

    def getActionProbability(self, state, action, *args, **kwargs):
        """
        Returns the probability that a specific action is chosen in a specific state by this policy.
        """
        #abstract

class RandomPolicy(Policy):
    """
    Returns a random action among all possible ones with a uniform distribution.
    """
    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def getAction(self, state, *args, **kwargs):
        possible_actions = self.environment.getPossibleActions(state)
        return np.random.choice(possible_actions)
    
class GreedyPolicy(Policy):
    """
    Returns the action with the highest Q-value for the current state.
    """
    def __init__(self, environment: Environment, q_table) -> None:
        self.environment = environment
        self.q_table = q_table

    def getAction(self, state, *args, **kwargs):
        possible_actions =  self.environment.getPossibleActions(state)
        q_values = {action: self.q_table[state][action] for action in possible_actions}
        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        return random.choice(best_actions)
    

    
    
