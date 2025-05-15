from agent import *
import random
from gui import Gui
from time import sleep

class Simulation():
    """
    A class that runs a simulation of the environment with two agents.
    """
    
    def __init__(self, environment: Environment, agentA: Agent, agentB: Agent, training = True, use_gui = True, mac=False) -> None:
        self.environment = environment
         #0 = player A, 1 = player B
        self.agentA = agentA
        self.agentB = agentB
        self.agentA.agent_index = 0
        self.agentB.agent_index = 1
        environment.reset()
        self.state = environment.getCurrentState()
        
        self.training = training
        self.mac = mac
        self.cooldown = 2
        if use_gui:
            self.gui = Gui(environment,mac)
            
            self.gui.run()
        else:
            self.gui = None

    def run(self, num_episodes: int = 1000):
        """
        Runs the simulation for a given number of episodes.
        """
        for episode in range(num_episodes):

            print("restarting episode", episode)
            self.environment.reset()
            self.state = self.environment.getCurrentState()
            done = False
            while not done:
                if self.gui and self.mac:
                    self.gui.run()
                if not self.mac or not self.training:
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
                if self.training:
                    self.agentA.value_function.updateQValue(self.state, 
                                                            next_state, 
                                                            actionA, 
                                                            actionB, 
                                                            self.environment.getPossibleActions(self.state, self.agentA.agent_index), 
                                                            self.environment.getPossibleActions(self.state, 1 - self.agentA.agent_index), 
                                                            rewardA)
                    self.agentB.value_function.updateQValue(self.state, 
                                                            next_state, 
                                                            actionB,
                                                            actionA,
                                                            self.environment.getPossibleActions(self.state, self.agentB.agent_index), 
                                                            self.environment.getPossibleActions(self.state, 1 - self.agentB.agent_index), 
                                                            rewardB)

                # Check if the episode is done
                if next_state[0][0] < 0 or next_state[1][0] > 4:
                    done = True
                
                # Update state
                self.state = next_state