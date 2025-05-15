from collections import defaultdict
from scipy.optimize import linprog
import random
import numpy as np

def state_to_tuple(state: list) -> tuple:
    """
    Converts a state from a list to a tuple.
    """
    return ((state[0][0], state[0][1]), (state[1][0], state[1][1]), state[2])

class Policy():

    def getAction(self, state, q_table, *args, **kwargs):
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
    def __init__(self, agent) -> None:
        self.agent = agent

    def getAction(self, state: list, possible_actions: list,q_table: dict = None) -> str:
        return np.random.choice(possible_actions)
    
class GreedyPolicy(Policy):
    """
    Returns the action with the highest Q-value for the current state.
    """
    def __init__(self, q_table, agent) -> None:
        self.q_table = q_table
        self.agent = agent

    def getAction(self, state: list, possible_actions: list,q_table: dict = None) -> str:
        state = state_to_tuple(state)
        q_values = {action: self.q_table[str(state)][action] for action in possible_actions}
        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        return random.choice(best_actions)
    

class EpsilonGreedyPolicy(Policy):
    """
    Exploits with probability epsilon or otherwise returns the action with the highest Q-value for the current state.
    """
    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon


    def getAction(self, state: list, possible_actions:list,q_table: dict) -> str:
        state = state_to_tuple(state)
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        # Exploitation
        q_values = {action: q_table[str(state)][action] for action in possible_actions}
        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        return random.choice(best_actions)
    
    

class LearnedMiniMaxPolicy(Policy):
    def __init__(self, environment, agent_idx, explore, pi=None) -> None:

        self.explore = explore
        self.pi = {}

        if pi is not None:
            self.pi = pi
        else:
            # Initialize the policy with uniform distribution over possible actions
            # for all states in the environment
            # Assuming the environment has a method to get all possible actions
            # for each state and agent
            
            for a in range(5):
                for b in range(4):
                    for c in range(5):
                        for d in range(4):
                            for e in [0, 1]:
                                state = ((a, b), (c, d), e)
                                actions = environment.getPossibleActions(state, agent_idx)
                                if actions:
                                    self.pi[state] = {action: 1 / len(actions) for action in actions}

    def getAction(self, state, possible_actions,q_table: dict = None) -> str:
        """
        Returns the action according to the policy based on the current state.
        """
        if random.random() < self.explore:
            return random.choice(possible_actions)
        state = state_to_tuple(state)
        if str(state) not in self.pi:
            return None
        actions = list(self.pi[str(state)].keys())
        probabilities = list(self.pi[str(state)].values())
        action = np.random.choice(actions, p=probabilities)
        if (action not in possible_actions):
            ValueError(f"Action {action} not in possible actions {possible_actions}")
        return action


    def update(self, state, possible_actions, possible_actions_opponent, Q_Function): 
        """
        Updates π[str(state)] using linear programming.
        """
        state = state_to_tuple(state)
        Q = Q_Function.Q[str(state)]
        num_actions = len(possible_actions)

        # Objective: maximize z (min value over opponent actions), so we minimize -z
        c = np.zeros(num_actions + 1)
        c[-1] = -1  # Coefficient for z

        # Constraints: z <= sum_a π[a] * Q[a][o]  →  -Q + z <= 0 for each opponent action o
        A_ub = []
        b_ub = []
        for o in possible_actions_opponent:
            constraint = [-Q[a][o] for a in possible_actions]  # -Q[a][o] * π[a]
            constraint.append(1)  # + z
            A_ub.append(constraint)
            b_ub.append(0)

        # Sum of π[a] = 1 (probability distribution)
        A_eq = [[1] * num_actions + [0]]
        b_eq = [1]

        # Bounds: π[a] in [0,1], z is unbounded
        bounds = [(0, 1)] * num_actions + [(None, None)]

        # Solve linear program
        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs"
        )

        if res.success:
            self.pi[str(state)] = {a: res.x[i] for i, a in enumerate(possible_actions)}
            z = res.x[-1]
            return z
        else:
            # fallback to uniform distribution
            self.pi[str(state)] = {a: 1 / len(possible_actions) for a in possible_actions}
            return 0.0
        
        

class MockPolicy(Policy):
    """
    A mock policy that does nothing.
    """
    def __init__(self, agent, environment) -> None:
        self.agent = agent
        self.environment = environment

    def getAction(self, state: list, possible_actions: list,q_table: dict = None) -> str:
        if self.environment.mock_actions[self.agent] == "":
            return "stay"
        else:
            old_action = self.environment.mock_actions[self.agent]
            self.environment.mock_actions[self.agent] = ""
            #print(old_action)
            return old_action
    
    def update(self):
        pass