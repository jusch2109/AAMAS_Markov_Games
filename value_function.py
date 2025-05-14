from collections import defaultdict
from policy import *

def state_to_tuple(state: list) -> tuple:
    """
    Converts a state from a list to a tuple.
    """
    return ((state[0][0], state[0][1]), (state[1][0], state[1][1]), state[2])


'''Missing: Policy for minimaxQ_Function and general Policy -> i have to look at this first'''

class Value_Function():

    def getValue(self, state, *args, **kwargs):
        """
        Returns the state-value of a given state.
        """
        #abstract

    def getQValue(self, state, action, *args, **kwargs):
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        #abstract

    def updateQValue(self, state, future_state, action, possible_actions: list, possible_opponent_actions: list,):
        """
        Updates the Q-value of a given state-action pair.
        """
        #abstract


class Q_Function(Value_Function):

    def __init__(self, policy: Policy, Q = None, start_value: int = 0.0, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """
        Initializes the value function with a default value.
        """
        if Q is not None:
            self.Q = Q
        else:
            # creates dict of form {state: {action: value}} 
            self.Q = defaultdict(lambda: defaultdict(lambda: start_value))

        self.start_value = start_value
        self.learning_rate = learning_rate
        self.policy = policy
        self.discount_factor = discount_factor
        self.policy = policy
        

    def getValue(self, state:list) -> float:
        """
        Returns the state-value of a given state.
        """
        state = state_to_tuple(state)
        return max(self.Q[state].values())

    def getQValue(self, state:list, action: str, action_opponent: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        state = state_to_tuple(state)
        return self.Q[state][action]
    
    def updateQValue(self, state: list, future_state: list, action: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        state = state_to_tuple(state)
        future_state = state_to_tuple(future_state)
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] +\
                                 self.learning_rate * (reward + self.discount_factor * self.getValue(future_state))

     
        
class MinimaxQ_Function(Value_Function):

    def __init__(self, policy: LearnedMiniMaxPolicy, Q: dict = None, V: dict = None, start_value: int = 0.0, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """
        Initializes the value function with a default value.
        """
        self.start_value = start_value
        if Q is not None:
            self.Q = Q
        else:
            # creates dict of form {state: {action: {opponent_action:value}}} 
            self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: start_value)))
        if V is not None:
            self.V = V
        else:
            # creates dict of form {state: value}
            self.V = defaultdict(lambda: start_value)
        self.start_value = start_value
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.policy = policy
        

    def getValue(self, state:list) -> float:
        """
        Returns the state-value of a given state.
        """
        state = state_to_tuple(state)
        if state not in self.V:
            return self.start_value
        
        return self.V[state]


    def getQValue(self, state:list, action: str, action_opponent: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        state = state_to_tuple(state)
        return self.Q[state][action][action_opponent]
    
    def updateQValue(self, state: list, future_state: list, action: str, action_opponent: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        state = state_to_tuple(state)
        future_state = state_to_tuple(future_state)
        self.Q[state][action][action_opponent] = (1 - self.learning_rate) * self.Q[state][action][action_opponent] + self.learning_rate * (reward + self.discount_factor * self.getValue(state))
        self.learning_rate = self.learning_rate * self.discount_factor
        
        self.V[state] = self.policy.update(state, possible_actions, possible_opponent_actions, self)


class RandomPolicy_Value_Function(Value_Function):
    """
    A random policy that selects actions uniformly at random.
    """
    def __init__(self, agent:int) -> None:
        self.policy = RandomPolicy(agent)

    def getValue(self, state: list) -> float:
        """
        Returns the state-value of a given state.
        """
        return 0.0
    def getQValue(self, state: list, action: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        return 0.0
    def updateQValue(self, state: list, future_state: list, action: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        pass
