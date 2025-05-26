import random
import numpy as np

class SoccerEnvironment:
  
  def __init__(self):
    """
    Initializes the environment
    """
    self.ball = random.randint(0, 1)
    #structure: position A, position B, ball
    #position A and B are tuples of (x,y) coordinates
    #ball is an integer representing the index of the player with the ball
    #0 = player A, 1 = player B
    self.state = [[3,2], [1,1], self.ball]  #  [3,2,1,1,1]
    self.mock_actions = ["",""]
        
  def getCurrentState(self):
    """
    Returns the current state of enviornment
    """
    return self.state
    
  def getPossibleActions(self, state: list, agent: int | str) -> list:
    """
    Returns possible actions the agent 
    can take in the given state. Can
    return the empty list if we are in 
    a terminal state.
    """

    # A
    if agent == 0 or agent == "A":
        posiible_actions_A = ['stay']
        if state[0][0] < 4:
            posiible_actions_A.append('move_right')
        if state[0][0] > 0 or (0 < state[0][1] < 3 and state[2] == 0):
            posiible_actions_A.append('move_left')
        if state[0][1] < 3:
            posiible_actions_A.append('move_up')
        if state[0][1] > 0:
            posiible_actions_A.append('move_down')
        return posiible_actions_A

    # B
    if agent == 1 or agent == " ":
        posiible_actions_B = ['stay']
        if state[1][0] < 4 or (0 < state[1][1] < 3 and state[2] == 1):
            posiible_actions_B.append('move_right')
        if state[1][0] > 0 :
            posiible_actions_B.append('move_left')
        if state[1][1] < 3:
            posiible_actions_B.append('move_up')
        if state[1][1] > 0:
            posiible_actions_B.append('move_down')
        return posiible_actions_B
    
    return []


                
  def doAction(self, action: str, agent: int | str):
    """
    Performs the given action in the current
    environment state and updates the enviornment.
    
    Returns a (reward, nextState) pair
    """

    #print(action)
    idx = 0
    if agent == 0 or agent == "A":
        idx = 0
    elif agent == 1 or agent == "B":
        idx = 1
    else:
        raise ValueError("Invalid agent")
    
    pos_old = self.state[idx]
    
    if action not in self.getPossibleActions(self.state, agent):
        print(self.state)
        print(action)
        raise ValueError("Invalid action for the given state")

    if action == 'move_right':
        self.state[idx][0] += 1
    elif action == 'move_left':
        self.state[idx][0] -= 1
    elif action == 'move_up':
        self.state[idx][1] += 1
    elif action == 'move_down':
        self.state[idx][1] -= 1
    elif action == 'stay':
        pass
    else:
        raise ValueError("Invalid action")
    
    if self.state[idx] == self.state[1-idx]:
        #if the two players are in the same position, the player with the ball
        #can pass it to the other player
        if self.ball == idx:
            self.ball = 1 - idx
            self.state[2] = self.ball
        ## undo move
        if action == 'move_right':
            self.state[idx][0] -= 1
        elif action == 'move_left':
            self.state[idx][0] += 1
        elif action == 'move_up':
            self.state[idx][1] -= 1
        elif action == 'move_down':
            self.state[idx][1] += 1

    reward = 0
    if (idx ==0 and self.ball == 0 and self.state[idx][0]<0 ) or (idx == 1 and self.ball == 1 and self.state[idx][0]>4):
        #if the player with the ball is in the same position as the other player
        #the player with the ball gets a reward of 1
        reward = 1

    return (reward, self.state)
        

        
  def reset(self):
    """
    Resets the current state to the start state
    """
    self.ball = random.randint(0, 1)
    #structure: position A, position B, ball
    #position A and B are tuples of (x,y) coordinates
    #ball is an integer representing the index of the player with the ball
    #0 = player A, 1 = player B
    self.state = [[3,2], [1,1], self.ball]


class CatchEnvironment:

    def __init__(self):
        """
        Initializes the environment
        """
        self.hunter = random.randint(0, 1)
        #structure: position A, position B, ball
        #position A and B are tuples of (x,y) coordinates
        #ball is an integer representing the index of the player with the ball
        #0 = player A, 1 = player B
        self.state = [[0,0], [3,3], self.hunter]  #  [3,2,1,1,1]
        self.mock_actions = ["",""]
            
    def getCurrentState(self):
        """
        Returns the current state of enviornment
        """
        return self.state
        
    def getPossibleActions(self, state: list, agent: int | str) -> list:
        """
        Returns possible actions the agent 
        can take in the given state. Can
        return the empty list if we are in 
        a terminal state.
        """

        # A
        if agent == 0 or agent == "A":
            possible_actions_A = ['stay']
            if state[0][0] < 3:
                possible_actions_A.append('move_right')
            if state[0][0] > 0:
                possible_actions_A.append('move_left')
            if state[0][1] < 3:
                possible_actions_A.append('move_up')
            if state[0][1] > 0:
                possible_actions_A.append('move_down')
            return possible_actions_A

        # B
        if agent == 1 or agent == " ":
            possible_actions_B = ['stay']
            if state[1][0] < 3:
                possible_actions_B.append('move_right')
            if state[1][0] > 0 :
                possible_actions_B.append('move_left')
            if state[1][1] < 3:
                possible_actions_B.append('move_up')
            if state[1][1] > 0:
                possible_actions_B.append('move_down')
            return possible_actions_B
        
        return []


                    
    def doAction(self, agentA, action_A: str, agentB, action_B: str):
        """
        Performs the given action in the current
        environment state and updates the enviornment.
        
        Returns a (reward, nextState) pair
        """

        
        if action_A not in self.getPossibleActions(self.state, agentA):
            raise ValueError("Invalid action for the given state")
        if action_B not in self.getPossibleActions(self.state, agentB):
            raise ValueError("Invalid action for the given state")

        if random.random() < 0.5:
            if action_A == 'move_right':
                self.state[agentA][0] += 1
            elif action_A == 'move_left':
                self.state[agentA][0] -= 1
            elif action_A == 'move_up':
                self.state[agentA][1] += 1
            elif action_A == 'move_down':
                self.state[agentA][1] -= 1
            elif action_A == 'stay':
                pass
            else:
                raise ValueError("Invalid action")
        
            if self.state[agentA] == self.state[agentB]:
                if self.hunter == agentA:
                    self.reset()
                    return (1,-1), self.state
                else:
                    self.reset()
                    return (-1,1), self.state
                
            if action_B == 'move_right':
                self.state[agentB][0] += 1
            elif action_B == 'move_left':
                self.state[agentB][0] -= 1
            elif action_B == 'move_up':
                self.state[agentB][1] += 1
            elif action_B == 'move_down':
                self.state[agentB][1] -= 1
            elif action_B == 'stay':
                pass
            else:
                raise ValueError("Invalid action")
        
            if self.state[agentA] == self.state[agentB]:
                if self.hunter == agentA:
                    self.reset()
                    return (1,-1), self.state
                else:
                    self.reset()
                    return (-1,1), self.state
        else:
            if action_B == 'move_right':
                self.state[agentB][0] += 1
            elif action_B == 'move_left':
                self.state[agentB][0] -= 1
            elif action_B == 'move_up':
                self.state[agentB][1] += 1
            elif action_B == 'move_down':
                self.state[agentB][1] -= 1
            elif action_B == 'stay':
                pass
            else:
                raise ValueError("Invalid action")
        
            if self.state[agentA] == self.state[agentB]:
                self.reset()
                if self.hunter == agentA:
                    return (1,-1), self.state
                else:
                    return (-1,1), self.state
                
            if action_A == 'move_right':
                self.state[agentA][0] += 1
            elif action_A == 'move_left':
                self.state[agentA][0] -= 1
            elif action_A == 'move_up':
                self.state[agentA][1] += 1
            elif action_A == 'move_down':
                self.state[agentA][1] -= 1
            elif action_A == 'stay':
                pass
            else:
                raise ValueError("Invalid action")
        
            if self.state[agentA] == self.state[agentB]:
                self.reset()
                if self.hunter == agentA:
                    return (1,-1), self.state
                else:
                    return (-1,1), self.state
            
            
        return (0,0), self.state
            

            
    def reset(self):
        """
        Resets the current state to the start state
        """
        self.hunter = random.randint(0, 1)
        #structure: position A, position B, ball
        #position A and B are tuples of (x,y) coordinates
        #ball is an integer representing the index of the player with the ball
        #0 = player A, 1 = player B
        self.state = [[0,0], [3,3], self.hunter]