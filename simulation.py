from agent import *
import random
from gui import Gui
from time import sleep

class Simulation():
    """
    A class that runs a simulation of the environment with two agents.
    """
    
    def __init__(self, environment: Environment, agentA: Agent, agentB: Agent, use_gui = True) -> None:
        self.environment = environment
         #0 = player A, 1 = player B
        self.agentA = agentA
        self.agentB = agentB
        self.agentA.agent_index = 0
        self.agentB.agent_index = 1
        self.state = environment.reset()
        self.training = True
        self.cooldown = 2
        if use_gui:
            self.gui = Gui(environment)
            self.gui.run()
        else:
            self.gui = None

    def run(self, num_episodes: int = 1000):
        """
        Runs the simulation for a given number of episodes.
        """
        for episode in range(num_episodes):
            self.environment.reset()
            self.state = self.environment.getCurrentState()
            done = False
            while not done:
                sleep(self.cooldown)
                actionA = self.agentA.getAction(self.state)
                actionB = self.agentB.getAction(self.state)
                print(self.environment.state)
                print(actionA, actionB)
                if random.random() < 0.5:
                    rewardA, next_state = self.environment.doAction(actionA, 0)
                    rewardB, next_state = self.environment.doAction(actionB, 1)
                else:
                    rewardB, next_state = self.environment.doAction(actionB, 1)
                    rewardA, next_state = self.environment.doAction(actionA, 0)
                
                # Update Q-values    TODO: probably has to be changed as we dont necessarily use Qlearning
                if self.training and False:
                    self.agentA.value_function.updateQValue(self.state, actionA, actionB, rewardA)
                    self.agentB.value_function.updateQValue(self.state, actionB, actionB, rewardB)

                # Check if the episode is done
                if next_state[0][0] < 0 or next_state[1][0] > 4:
                    done = True
                
                # Update state
                self.state = next_state