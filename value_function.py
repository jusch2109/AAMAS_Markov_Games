from collections import defaultdict

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


class Q_Function(Value_Function):

    def __init__(self, Q = None, start_value: int = 0.0):
        """
        Initializes the value function with a default value.
        """
        if Q is not None:
            self.Q = Q
        else:
            # creates dict of form {state: {action: value}} 
            self.Q = defaultdict(lambda: defaultdict(lambda: start_value))

        

    def getValue(self, state:list) -> float:
        """
        Returns the state-value of a given state.
        """
        return max(self.Q[state].valiues())

    def getQValue(self, state:list, action: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        return self.Q[state][action]
    
    def updateQValue(self, state: list, action: str, action_opponent: str, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        pass
        
class minimaxQ_Function(Value_Function):

    def __init__(self, Q = None, start_value: int = 0.0):
        """
        Initializes the value function with a default value.
        """
        if Q is not None:
            self.Q = Q
        else:
            # creates dict of form {state: {action: value}} 
            self.Q = defaultdict(lambda: defaultdict(lambda: start_value))
        

    def getValue(self, state:list) -> float:
        """
        Returns the state-value of a given state.
        """
        return max(self.Q[state].valiues())

    def getQValue(self, state:list, action: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        return self.Q[state][action]
    
    def updateQValue(self, state: list, action: str, action_opponent: str, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        pass