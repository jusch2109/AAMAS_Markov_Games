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


class TicTacToeEnvironment:

    def __init__(self):
        """
        Initializes the environment
        """
        
        self.current_player = np.random.randint(0, 2)
        self.state = np.zeros((3,3))
        self.done = False

    def getCurrentState(self):
        """
        Returns the current state of enviornment
        """
        return self.state, self.current_player
    
    def getPossibleActions(self) -> list:
        """
        Returns possible actions the agent 
        can take in the given state. Can
        return the empty list if we are in 
        a terminal state.
        """
        if self.done:
            return []
        
        possible_actions = []
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    possible_actions.append((i,j))
        
        return possible_actions
    
    def doAction(self, action: tuple, agent: int ):
        """
        Performs the given action in the current
        environment state and updates the enviornment.
        
        Returns a (reward, nextState) pair
        """

        if action not in self.getPossibleActions():
            raise ValueError("Invalid action for the given state")

        self.state[action[0]][action[1]] = agent + 1

        reward = 0
        if self.check_winner(self.current_player):
            reward = 1
            self.done = True


        self.current_player = 1 - self.current_player

        return (reward, self.state, self.done)
    
    def check_winner(self, player):
        """
        Checks if the given player has won the game
        """
        # Check rows
        for row in self.state:
            if np.all(row == player + 1):
                return True

        # Check columns
        for col in self.state.T:
            if np.all(col == player + 1):
                return True

        # Check diagonals
        if np.all(np.diag(self.state) == player + 1) or np.all(np.diag(np.fliplr(self.state)) == player + 1):
            return True

        return False
    
    def reset(self):
        """
        Resets the current state to the start state
        """
        self.current_player = np.random.randint(0, 2)
        self.state = np.zeros((3,3))
        self.done = False