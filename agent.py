from policy import Policy
from value_function import ValueFunction

class Agent():
    """
    An agent that interacts with the environment.
    """

    def __init__(self, policy: Policy, value_function: ValueFunction) -> None:
        self.policy = policy
        self.value_function = value_function

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