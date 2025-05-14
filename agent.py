from policy import Policy
from value_function import Value_Function

class Agent():
    """
    An agent that interacts with the environment.
    """
    #0 = player A, 1 = player B   if not defined -> -1
    def __init__(self, policy: Policy, value_function: Value_Function, agent_index = -1) -> None:
        self.policy = policy
        self.value_function = value_function
        self.agent_index = agent_index  

    def getAction(self, state, *args, **kwargs):
        """
        Returns an action according to the policy based on the current state
        """
        return self.policy.getAction(state, *args, **kwargs)

    def getValue(self, state, *args, **kwargs):
        """
        Returns the state-value of a given state.
        """
        return self.value_function.getValue(state, *args, **kwargs)
    
    def getQValue(self, state, action, *args, **kwargs):
        """
        Returns the Q-value of a given state-action pair.
        """
        return self.value_function.getQValue(state, action, *args, **kwargs)