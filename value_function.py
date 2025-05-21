from collections import defaultdict
from policy import *
import json
import os
from environment import SoccerEnvironment

def state_to_tuple(state: list) -> tuple:
    """
    Converts a state from a list to a tuple.
    """
    if type(state) != tuple:
        return ((state[0][0], state[0][1]), (state[1][0], state[1][1]), state[2])
    else:
        return state


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

    def getAction(self, state, possible_actions, *args, **kwargs):
        """
        Returns an action according to the policy based on the current state
        """
        #abstract

    def updateQValue(self, state, future_state, action, action_opponent: str, possible_actions: list, possible_opponent_actions: list, reward:int):
        """
        Updates the Q-value of a given state-action pair.
        """
        #abstract
    
    def save_dict(self, filename: str):
        """
        Saves the Q-value dictionary to a file.
        """
        #abstract
    def load_dict(self, filename: str):
        """
        Loads the Q-value dictionary from a file.
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
            self.Q["training_episodes"] = 0
        self.start_value = start_value
        self.learning_rate = learning_rate
        self.policy = policy
        self.discount_factor = discount_factor
        

    def getAction(self, state:list, possible_actions:list) -> str:
        """
        Returns an action according to the policy based on the current state
        """
        state = state_to_tuple(state)
        return self.policy.getAction(state, possible_actions, self.Q)

    def getValue(self, state:list) -> float:
        """
        Returns the state-value of a given state.
        """
        state = state_to_tuple(state)
        if str(state) not in self.Q or self.Q[str(state)] == {}:
            #print(self.Q[str(state)])
            #print("\n\n")
            return self.start_value
        return max(self.Q[str(state)].values())

    def getQValue(self, state:list, action: str, action_opponent: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        state = state_to_tuple(state)
        return self.Q[str(state)].get(action,self.start_value)
    
    def updateQValue(self, state: list, future_state: list, action: str, action_opponent: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """

        state = state_to_tuple(state)
        future_state = state_to_tuple(future_state)
        #print(self.Q[str(state)])
        #print("\n")
        self.Q[str(state)][action] = (1 - self.learning_rate) * self.Q[str(state)].get(action,self.start_value) +\
                                 self.learning_rate * (reward + self.discount_factor * self.getValue(future_state))
        #if(self.getValue(future_state) != 0):
            #print(future_state)
            #print(state)
            #print(self.Q[str(state)])


    def save_dict(self, filename="q_table"):
        if not filename.endswith(".json"):
            filename = filename + ".json"
        out_file = open(filename, "w")
        json.dump(self.Q, out_file, indent = 4)

    def load_dict(self, filename="q_table"):
        if not filename.endswith(".json"):
            filename = filename + ".json"
        if not os.path.exists(filename):
            print("File does not exist")
            return
        # Opening JSON file
        f = open(filename,)
        
        # returns JSON object as 
        # a dictionary
        l = json.load(f)
        
        self.Q = defaultdict(lambda: defaultdict(lambda: self.start_value), l)
        self.policy.q_table = self.Q
        
     

class JAL_AM_Q_Function(Value_Function):

    def __init__(self, policy: Policy, Q = None, opponent_model = None, env = None, agent_idx = None, start_value: int = 0.0, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """
        Initializes the value function with a default value.
        """
        if Q is not None:
            self.Q = Q
        else:
            # creates dict of form {state: {action: {opponent_action: value}}} 
            self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: start_value)))
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        for d in range(4):
                            for e in [0, 1]:
                                state = ((a, b), (c, d), e)
                                for action in env.getPossibleActions(state, agent_idx):
                                    for opponent_action in env.getPossibleActions(state, 1 - agent_idx):
                                        if  state[1]==(3,3) and opponent_action == "move_up":
                                            start_value = 0
                                        self.Q[str(state_to_tuple(state))][action][opponent_action] = start_value
                                       
        if opponent_model is not None:
            self.opponent_model = opponent_model
        else:
            # creates dict of form {state: {action: value}} 
            self.opponent_model = defaultdict(lambda: defaultdict(lambda: 0.0))
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        for d in range(4):
                            for e in [0, 1]:
                                state = ((a, b), (c, d), e)
                                actions = env.getPossibleActions(state, 1 - agent_idx)
                                if actions:
                                    self.opponent_model[str(state_to_tuple(state))] = {action: 1 / len(actions) for action in actions}
        
        # create dict of form {state: {action: value}}
        self.opponent_counter = defaultdict(lambda: defaultdict(lambda: 0))

        # create dict of form {state: {action: value}}
        self.AV = defaultdict(lambda: defaultdict(lambda: 0))
        self.update_AV()

        self.start_value = start_value
        self.learning_rate = learning_rate
        self.policy = policy
        self.discount_factor = discount_factor
        

    def getAction(self, state:list, possible_actions:list) -> str:
        """
        Returns an action according to the policy based on the current state
        """
        state = state_to_tuple(state)
        return self.policy.getAction(state, possible_actions, self.AV)

    def getValue(self, state:list) -> float:
        """
        Returns the state-value of a given state.
        """
        state = state_to_tuple(state)

        return max(self.AV[str(state)].values())

    def getQValue(self, state:list, action: str, action_opponent: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        state = state_to_tuple(state)
        return self.Q[str(state)][action][action_opponent]
    
    def updateQValue(self, state: list, future_state: list, action: str, action_opponent: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """

        state = state_to_tuple(state)
        future_state = state_to_tuple(future_state)
        
        # update the opponent counter and model
        self.opponent_counter[str(state)][action_opponent] += 1
        self.opponent_model[str(state)][action_opponent] = self.opponent_counter[str(state)][action_opponent] / sum(self.opponent_counter[str(state)].values())

        # update the action value
        self.Q[str(state)][action][action_opponent] = (1 - self.learning_rate) * self.Q[str(state)][action][action_opponent] +\
                                 self.learning_rate * (reward + self.discount_factor * (max(self.AV[str(future_state)].values()) - self.Q[str(state)][action][action_opponent]))
        
        # update value function
        self.update_AV()

        self.learning_rate = self.learning_rate * self.discount_factor


    def update_AV(self):
        self.AV = defaultdict(lambda: defaultdict(lambda: 0))
        for state in self.Q.keys():
            for action in self.Q[state].keys():
                for action_opponent in self.Q[state][action].keys():
                    self.AV[state][action] += self.Q[state][action][action_opponent]*self.opponent_model[state][action_opponent]

    def save_dict(self, filename="q_table"):

        out_file = open(filename + "_Q.json", "w")
        json.dump(self.Q, out_file, indent = 4)
        out_file = open(filename + "_opponent_model.json", "w")
        json.dump(self.opponent_model, out_file, indent = 4)

    def load_dict(self, filename="min_max"):
        
        # Opening JSON file
        f_Q = open(filename + '_Q.json',)
        f_opponent = open(filename + '_opponent_model.json',)

        if not os.path.exists(f_Q.name) or not os.path.exists(f_opponent.name):
            return
        
        # returns JSON object as 
        # a dictionary
        self.Q = json.load(f_Q)
        self.opponent_model = json.load(f_opponent)
        return
        
        
class MinimaxQ_Function(Value_Function):

    def __init__(self, policy: LearnedMiniMaxPolicy, Q: dict = None, V: dict = None, start_value: int = 0.0, learning_rate: float = 0.1, discount_factor: float = 0.9, decay: float = 0.9999):
        """
        Initializes the value function with a default value.
        """
        self.start_value = start_value
        if Q is not None:
            self.Q = Q
        else:
            # creates dict of form {state: {action: {opponent_action:value}}} 
            self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: start_value)))
            self.Q["training_episodes"] = 0
        if V is not None:
            self.V = V
        else:
            # creates dict of form {state: value}
            self.V = defaultdict(lambda: start_value)
        self.start_value = start_value
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay = decay
        self.policy = policy
        
    def getAction(self, state:list, possible_actions:list) -> str:
        """
        Returns an action according to the policy based on the current state
        """
        state = state_to_tuple(state)
        return self.policy.getAction(state, possible_actions, self.Q)

    def getValue(self, state:list) -> float:
        """
        Returns the state-value of a given state.
        """
        if type(state) == list:
            state = state_to_tuple(state)
        if state not in self.V:
            return self.start_value
        
        return self.V[str(state)]


    def getQValue(self, state:list, action: str, action_opponent: str) -> float:
        """
        returns the state-action-value or Q-Value of a given state-action pair.
        """
        state = state_to_tuple(state)
        return self.Q[str(state)][action][action_opponent]
    
    def updateQValue(self, state: list, future_state: list, action: str, action_opponent: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        state = state_to_tuple(state)
        future_state = state_to_tuple(future_state)
        self.Q[str(state)][action][action_opponent] = (1 - self.learning_rate) * self.Q[str(state)][action][action_opponent] + self.learning_rate * (reward + self.discount_factor * self.getValue(str(future_state)))
        self.learning_rate = self.learning_rate * self.decay
        
        self.V[str(state)] = self.policy.update(state, possible_actions, possible_opponent_actions, self.Q)

    def save_dict(self, filename="min_max"):

        out_file = open(filename + '_Q.json', "w")
        json.dump(self.Q, out_file, indent = 4)
        out_file = open(filename + '_V.json', "w")
        json.dump(self.V, out_file, indent = 4)

    def load_dict(self, filename="min_max"):
        
        # Opening JSON file
        f_Q = open(filename + '_Q.json',)
        f_V = open(filename + '_V.json',)
        if not os.path.exists(f_Q.name) or not os.path.exists(f_V.name):
            return
        
        # returns JSON object as 
        # a dictionary
        self.Q = json.load(f_Q)
        self.V = json.load(f_V)
        return

class RandomPolicy_Value_Function(Value_Function):
    """
    A random policy that selects actions uniformly at random.
    """
    def __init__(self, agent:int) -> None:
        self.policy = RandomPolicy(agent)
        self.Q = None

    def getAction(self, state: list, possible_actions: list) -> str:
        """
        Returns an action according to the policy based on the current state
        """
        state = state_to_tuple(state)
        return self.policy.getAction(state, possible_actions, self.Q)

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
    def updateQValue(self, state: list, future_state: list, action: str, action_opponent: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        pass

    def save_dict(self, filename):
        pass

    def load_dict(self, filename):
        pass

class Mock_Value_Function(Value_Function):
    """
    A mock value function for testing purposes.
    """
    def __init__(self, agent, env) -> None:
        self.policy = MockPolicy(agent, env)
        self.Q = None

    def getAction(self, state: list, possible_actions: list) -> str:
        """
        Returns an action according to the policy based on the current state
        """
        state = state_to_tuple(state)
        return self.policy.getAction(state, possible_actions, self.Q)

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
    def updateQValue(self, state: list, future_state: list, action: str, action_opponent: str, possible_actions: list, possible_opponent_actions: list, reward: int):
        """
        Updates the Q-value of a given state-action pair.
        """
        pass
    
    def save_dict(self, filename):
        pass

    def load_dict(self, filename):
        pass
