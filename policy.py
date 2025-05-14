from collections import defaultdict
from policy import LearnedMiniMaxPolicy
from scipy.optimize import linprog
import random
import numpy as np

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
    def __init__(self, agent) -> None:
        self.agent = agent

    def getAction(self, state: list, possible_actions: list) -> str:
        return np.random.choice(possible_actions)
    
class GreedyPolicy(Policy):
    """
    Returns the action with the highest Q-value for the current state.
    """
    def __init__(self, q_table, agent) -> None:
        self.q_table = q_table
        self.agent = agent

    def getAction(self, state: list, possible_actions: list) -> str:
        q_values = {action: self.q_table[state][action] for action in possible_actions}
        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        return random.choice(best_actions)
    
class EpsilonGreedyPolicy(Policy):
    """
    Exploits with probability epsilon or otherwise returns the action with the highest Q-value for the current state.
    """
    def __init__(self, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon


    def getAction(self, state: list, q_table: dict, possible_actions:list) -> str:

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        # Exploitation
        q_values = {action: q_table[state][action] for action in possible_actions}
        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        return random.choice(best_actions)
    
    

class LearnedMiniMaxPolicy(Policy):
    def __init__(self, environment) -> None:

        self.pi = {}
        # Initialisiere gleichverteilte Policy
        for a in range(5):
            for b in range(4):
                for c in range(5):
                    for d in range(4):
                        for e in [0, 1]:
                            state = [(a, b), (c, d), e]
                            actions = environment.getPossibleActions(state)
                            if actions:
                                self.pi[state] = {action: 1 / len(actions) for action in actions}

    def getAction(self, state, possible_actions) -> str:
        """
        Returns the action according to the policy based on the current state.
        """
        if state not in self.pi:
            return None
        actions = list(self.pi[state].keys())
        probabilities = list(self.pi[state].values())
        action = np.random.choice(actions, p=probabilities)
        if (action not in possible_actions):
            ValueError(f"Action {action} not in possible actions {possible_actions}")
        return action


    def update(self, state, possible_actions, possible_actions_opponent, Q_Function): 
        """
        Updates π[state] using linear programming.
        """
        Q = Q_Function.Q[state]
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
            self.pi[state] = {a: res.x[i] for i, a in enumerate(possible_actions)}
            z = res.x[-1]
            self.minimaxQ_Function.V[state] = z
        else:
            # fallback to uniform distribution
            self.pi[state] = {a: 1 / len(possible_actions) for a in possible_actions}