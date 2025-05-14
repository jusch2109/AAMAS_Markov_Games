from value_function import Value_Function
from environment import Environment

class Agent():
    """
    An agent that interacts with the environment.
    """
    #0 = player A, 1 = player B   if not defined -> -1
    def __init__(self, environment: Environment, value_function: Value_Function, agent_index = -1) -> None:
        self.value_function = value_function
        self.agent_index = agent_index  
        self.environment = environment

    def getAction(self, state):
        """
        Returns an action according to the policy based on the current state
        """
        return self.value_function.policy.getAction(state, self.value_function.Q, self.environment.getPossibleActions(state, self.agent_index))

    def getValue(self, state):
        """
        Returns the state-value of a given state.
        """
        return self.value_function.getValue(state)
    
    def getQValue(self, state, action):
        """
        Returns the Q-value of a given state-action pair.
        """
        return self.value_function.getQValue(state, action)
    

    def updateQValue(self, state, future_state, action, reward):
        """
        Updates the Q-value of a given state-action pair.
        """
        self.value_function.updateQValue(state, future_state, 
                                         action, 
                                         self.environment.getPossibleActions(self.agent_index), 
                                         self.environment.getPossibleActions(1-self.agent_index), 
                                         reward)