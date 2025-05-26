from collections import defaultdict
from scipy.optimize import linprog
import random
import numpy as np
import json
import os

def state_to_tuple(state: list) -> tuple:
    """
    Converts a state from a list to a tuple.
    """
    if type(state) != tuple:
        return ((state[0][0], state[0][1]), (state[1][0], state[1][1]), state[2])
    else:
        return state

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
        ## this kind of has a problem that it assumes starting value has to be 0. since it is always 0 its fine
        q_values = {action: self.q_table[str(state)].get(action,0) for action in possible_actions}

        max_value = max(q_values.values())
        best_actions = [a for a, v in q_values.items() if v == max_value]
        return random.choice(best_actions)
    
    def save_dict(self, filename="Pi_min_max"):
        pass
    
    def load_dict(self,filename = "a"):
        pass

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
    
    def save_dict(self, filename="Pi_min_max"):
        pass
        
    def load_dict(self,filename = "a"):
        pass


class QPolicy(Policy):
    """
    Returns the action with the highest Q-value for the current state.
    """
    def __init__(self, q_table, agent, epsilon = 0) -> None:
        self.epsilon = epsilon
        self.q_table = q_table
        self.agent = agent

    def getAction(self, state: list, possible_actions: list, q_table: dict = None) -> str:
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        state = state_to_tuple(state)
        q_values = {action: self.q_table[str(state)].get(action, 0) for action in possible_actions}

        min_q = min(q_values.values())   # this shifts values to [0:max_value]
        shifted_q = {a: q - min_q for a, q in q_values.items()}
        total = sum(shifted_q.values())

        if total == 0: # if all values are the same
            probabilities = [1.0 / len(possible_actions)] * len(possible_actions)
        else:
            probabilities = [q / total for q in shifted_q]

        action = np.random.choice(possible_actions, p=probabilities)
        return action
    
    def save_dict(self, filename="Pi_min_max"):
        pass
    
    def load_dict(self,filename = "a"):
        pass


class LearnedMiniMaxPolicy(Policy):
    def __init__(self, environment, agent_idx, explore, pi=None) -> None:

        self.explore = explore
        self.pi = {}
        # Initialisiere gleichverteilte Policy

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
                                    self.pi[str(state)] = {action: 1 / len(actions) for action in actions}

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
        If multiple optimal policies exist, prefers the one closest to a uniform distribution.
        """
    
        
        if type(state) != tuple:
            state = state_to_tuple(state)

        Q = Q_Function[str(state)]
        num_actions = len(possible_actions)

        # Stage 1: Maximize z (min over opponent actions)
        c = np.zeros(num_actions + 1)
        c[-1] = -1  # maximize z → minimize -z

        A_ub = []
        b_ub = []
        for o in possible_actions_opponent:
            constraint = [-Q[a][o] for a in possible_actions]
            constraint.append(1)  # + z
            A_ub.append(constraint)
            b_ub.append(0)

        A_eq = [[1] * num_actions + [0]]  # sum of π = 1
        b_eq = [1]
        bounds = [(0, 1)] * num_actions + [(None, None)]

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
            pi_values = res.x[:num_actions]
            # solve numerical issues
            pi_values = np.clip(pi_values, 0, 1)
            pi_values /= np.sum(pi_values)  
            
            self.pi[str(state)] = {a: pi_values[i] for i, a in enumerate(possible_actions)}
            return res.x[-1]
        else:
            self.pi[str(state)] = {a: 1 / len(possible_actions) for a in possible_actions}
            return 0.0
        
        
    def save_dict(self, filename="Pi_min_max"):
        if not filename.endswith(".json"):
            filename = filename + ".json"
        out_file = open(filename, "w")
        json.dump(self.pi, out_file, indent = 4)

    def load_dict(self, filename="Pi_min_max"):
        if not filename.endswith(".json"):
            filename = filename + ".json"
        if not os.path.exists(filename):
            return
        # Opening JSON file
        f = open(filename, )
        
        # returns JSON object as 
        # a dictionary
        self.pi = json.load(f)
        return
    


class JAL_AM_Policy(Policy):
    """
    Exploits with probability epsilon or otherwise returns the action with the highest Q-value for the current state.
    """
    def __init__(self, epsilon: float = 0.1, decay = 0.99999) -> None:
        self.epsilon = epsilon
        self.decay = decay


    def getAction(self, state: list, possible_actions:list, av_table: dict) -> str:
        state = state_to_tuple(state)
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        # Exploitation
        av_values = {action: av_table[str(state)][action] for action in possible_actions}
        max_value = max(av_values.values())
        best_actions = [a for a, v in av_values.items() if v == max_value]

        self.epsilon = self.epsilon * self.decay

        return random.choice(best_actions)

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
            #self.environment.mock_actions[self.agent] = ""
            ##print(old_action)
            return old_action
    
    def update(self):
        pass

class HandcraftedPolicy(Policy):
    """
    A policy that uses handcrafted rules to determine the action.
    """

    def __init__(self, agent,is_soccer) -> None:
        self.agent = agent
        self.is_soccer = is_soccer
        self.finta = None

    def getAction(self, state: list, possible_actions: list,q_table: dict = None) -> str:
        """
        Returns an action based on handcrafted rules.
        """
        if self.is_soccer:

            if state[2] == self.agent:
                # if we are attacking
                if self.agent == 1:
                    goals = [[5,1],[5,2]]
                else:
                    goals = [[-1,1],[-1,2]]

                if abs(state[self.agent][0] - state[1-self.agent][0]) + abs(state[self.agent][1] - state[1-self.agent][1]) > 2: # if distance is greater than 2 its safe to move to objective
                    self.finta = None
                    print("hes far")
                    return "move_right" if state[self.agent][0] < goals[0][0] else "move_left"
                if state[1-self.agent][1] == state[self.agent][1]:  # if we are in the same row
                    if abs(state[1-self.agent][0] - goals[0][0]) <  abs(state[self.agent][0] - goals[0][0]):  ## if he is closer to the goal
                        print("he is closer to the goal")
                        if self.finta is None:
                            self.finta = state[self.agent][1]
                            if self.finta == 1:
                                return "move_up"
                            else:
                                return "move_down"
                        else:
                            ## if we tried to trick him and he moved with us stay so he moves away
                            self.finta = None
                            return "stay"
                    else:
                        # if we are closer to the goal we just want to move to the goal
                        print("we are closer to the goal")
                        self.finta = None
                        return "move_right" if state[self.agent][0] < goals[0][0] else "move_left"
                else: # we are close but not on the same row, but maybe well just try to move to the goal
                    self.finta = None
                    print("we are close but not on the same row, but maybe well just try to move to the goal")
                    return "move_right" if state[self.agent][0] < goals[0][0] else "move_left"
            else:
                # if we are defending:

                # if we are on the same row we wanna be 1 square between them and the goal
                if state[1-self.agent][1] == state[self.agent][1]:
                    if self.agent == 1:
                        goal = state[1-self.agent][0] - 1
                    else:
                        goal = state[1-self.agent][0] + 1
                    if state[self.agent][0] < goal:
                        return "move_right"
                    elif state[self.agent][0] > goal:
                        return "move_left"
                    else:
                        return "stay"
                else:
                    # if he is in the sides we just want to be on "top" of him
                    if state[1-self.agent][1] > 2 or state[1-self.agent][1] < 1:
                        goal = state[1-self.agent][0]
                        if state[self.agent][0] < goal:
                            return "move_right"
                        elif state[self.agent][0] > goal:
                            return "move_left"
                        else:
                            return "stay"
                    else:
                        # if he is in the middle we want to be on the same row as him
                        if state[self.agent][1] < state[1-self.agent][1]:
                            return "move_up"
                        elif state[self.agent][1] > state[1-self.agent][1]:
                            return "move_down"
                        else:
                            return "stay"
        else:
            agent_pos = state[self.agent]
            opponent_pos = state[1 - self.agent]
                
            actions = possible_actions
            possibilities = []
            if state[2] == self.agent: # if we are catching

                dist_x = abs(opponent_pos[0] - agent_pos[0])
                dist_y = abs(opponent_pos[1] - agent_pos[1])
                if dist_x == dist_y: # if we are closer in x direction
                    if opponent_pos[0] > agent_pos[0]: # if opponent is to the right
                        possibilities.append("move_right")
                    elif opponent_pos[0] < agent_pos[0]: # if opponent is to the left
                        possibilities.append("move_left")
                    if opponent_pos[1] > agent_pos[1]:
                        possibilities.append("move_up")
                    elif opponent_pos[1] < agent_pos[1]:
                        possibilities.append("move_down")
                elif dist_x > dist_y: # if we are closer in y direction
                    if opponent_pos[0] > agent_pos[0]: # if opponent is to the right
                        possibilities.append("move_right")
                    elif opponent_pos[0] < agent_pos[0]: # if opponent is to the left
                        possibilities.append("move_left")
                else:
                    if opponent_pos[1] > agent_pos[1]:
                        possibilities.append("move_up")
                    elif opponent_pos[1] < agent_pos[1]:
                        possibilities.append("move_down")
            else: # if we are running away
                if opponent_pos[0] > agent_pos[0]: # if opponent is to the right
                    possibilities.append("move_left")
                elif opponent_pos[0] < agent_pos[0]: # if opponent is to the left
                    possibilities.append("move_right")
                if opponent_pos[1] > agent_pos[1]:
                    possibilities.append("move_down")
                elif opponent_pos[1] < agent_pos[1]:
                    possibilities.append("move_up")
                
                if opponent_pos[0] == agent_pos[0]:
                    possibilities.append("move_left")
                    possibilities.append("move_right")
                if opponent_pos[1] == agent_pos[1]:
                    possibilities.append("move_up")
                    possibilities.append("move_down")
            actions = [action for action in actions if action in possibilities]

            if actions == []:
                return "stay"
            return np.random.choice(actions)
    
    