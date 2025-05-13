from agent import *
from environment import *
from policy import *
from value_function import *
import random

class Simulation():
    """
    A class that runs a simulation of the environment with two agents.
    """
    
    def __init__(self, environment: Environment, agentA: Agent, agentB: Agent) -> None:
        self.environment = environment
        self.agentA = agentA
        self.agentB = agentB
        self.state = environment.reset()
        training = True


    def run(self, num_episodes: int = 1000):
        """
        Runs the simulation for a given number of episodes.
        """
        for episode in range(num_episodes):
            self.state = self.environment.reset()
            done = False
            while not done:
                actionA = self.agentA.getAction(self.state)
                actionB = self.agentB.getAction(self.state)
                
                if random.random() < 0.5:
                    rewardA, next_state = self.environment.doAction(actionA, 0)
                    rewardB, next_state = self.environment.doAction(actionB, 1)
                else:
                    rewardB, next_state = self.environment.doAction(actionB, 1)
                    rewardA, next_state = self.environment.doAction(actionA, 0)
                
                # Update Q-values
                if self.training:
                    self.agentA.value_function.updateQValue(self.state, actionA, actionB, rewardA)
                    self.agentB.value_function.updateQValue(self.state, actionB, actionB, rewardB)
                
                # Check if the episode is done
                if next_state[0][0] < 0 or next_state[1][0] > 4:
                    done = True
                
                # Update state
                self.state = next_state