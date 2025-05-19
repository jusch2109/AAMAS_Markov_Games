from agent import *
from policy import *
import random
from gui import Gui
from time import sleep
from tqdm import tqdm
import copy
from environment import *

class SoccerSimulation():
    """
    A class that runs a simulation of the environment with two agents.
    """
    
    def __init__(self, environment: SoccerEnvironment, agentA: Agent, agentB: Agent, explore_decay:int = 0.9, training = True, use_gui = True, mac=False) -> None:
        self.environment = environment
         #0 = player A, 1 = player B
        self.agentA = agentA
        self.agentB = agentB
        self.agentA.agent_index = 0
        self.agentB.agent_index = 1
        #environment.reset()
        self.state = environment.getCurrentState()

        self.explore_decay = explore_decay
        
        self.training = training
        self.mac = mac
        self.save_and_load = False
        self.cooldown = 2
        if use_gui:
            self.gui = Gui(environment,mac)
            
            self.gui.run()
        else:
            self.gui = None

    def run(self, num_episodes: int = 50000):
        """
        Runs the simulation for a given number of episodes.
        """
        A_wins = 0
        B_wins = 0
        if self.save_and_load:
            self.agentA.value_function.load_dict("q_A")
            self.agentB.value_function.load_dict("q_B")
        for episode in tqdm(range(num_episodes)):

            #print("restarting episode", episode)
            self.state = self.environment.getCurrentState()
            done = False
            while not done:
                if self.gui and self.mac:
                    self.gui.run()
                if not self.training and self.gui:
                    print("sleeping")
                    sleep(self.cooldown)

                actionA = self.agentA.getAction(self.state, self.environment.getPossibleActions(self.state, self.agentA.agent_index))
                actionB = self.agentB.getAction(self.state, self.environment.getPossibleActions(self.state, self.agentB.agent_index))
                #print(self.environment.state)
                #print(actionA, actionB)

                previous_state = copy.deepcopy(self.state) #[list(self.state[0]), list(self.state[1]), self.state[2]]   # copy with different reference

                if random.random() < 0.5:
                    rewardA, next_state = self.environment.doAction(actionA, 0)
                    rewardB, next_state = self.environment.doAction(actionB, 1)
                else:
                    rewardB, next_state = self.environment.doAction(actionB, 1)
                    rewardA, next_state = self.environment.doAction(actionA, 0)
                
                # Update Q-values    TODO: figure out if previous state should be the state before actions or the state before the actual action of the agent
                if self.training:
                    self.agentA.updateQValue(previous_state, 
                                                            next_state, 
                                                            actionA, 
                                                            actionB, 
                                                            self.environment.getPossibleActions(previous_state, self.agentA.agent_index), 
                                                            self.environment.getPossibleActions(previous_state, 1 - self.agentA.agent_index), 
                                                            rewardA)
                    self.agentB.updateQValue(previous_state, 
                                                            next_state, 
                                                            actionB,
                                                            actionA,
                                                            self.environment.getPossibleActions(previous_state, self.agentB.agent_index), 
                                                            self.environment.getPossibleActions(previous_state, 1 - self.agentB.agent_index), 
                                                            rewardB)

                # Check if the episode is done
                if next_state[0][0] < 0 or next_state[1][0] > 4:
                    done = True
                    self.environment.reset()
                
                if next_state[0][0] < 0:
                    A_wins += 1
                if next_state[1][0] > 4:
                    B_wins += 1


                # Update state
                self.state = next_state

            
                if self.training:
                    #self.agentA.value_function.Q["training_episodes"] += 1
                    #self.agentB.value_function.Q["training_episodes"] += 1
                            # Update exploration decay
                    if type(self.agentA.value_function.policy) == EpsilonGreedyPolicy:
                        self.agentA.value_function.policy.epsilon *= self.explore_decay
                    if type(self.agentB.value_function.policy) == EpsilonGreedyPolicy:
                        self.agentB.value_function.policy.epsilon *= self.explore_decay
                    if type(self.agentA.value_function.policy) == LearnedMiniMaxPolicy:
                        self.agentA.value_function.policy.explore *= self.explore_decay
                    if type(self.agentB.value_function.policy) == LearnedMiniMaxPolicy:
                        self.agentB.value_function.policy.explore *= self.explore_decay

        if self.training and self.save_and_load:
            self.agentA.value_function.save_dict()
            self.agentB.value_function.save_dict()
        print("A wins:", A_wins)
        print("B wins:", B_wins)


'''class TicTacToeSimulation():
    """
    A class that runs a simulation of the tictactoe environment with two agents.
    """
    def __init__(self, environment: TicTacToeEnvironment, agentA: Agent, agentB: Agent, explore_decay:int = 0.9, training = True) -> None:
        self.environment = environment
         #0 = player A, 1 = player B
        self.agentA = agentA
        self.agentB = agentB
        self.agentA.agent_index = 0
        self.agentB.agent_index = 1
        #environment.reset()
        self.state = environment.getCurrentState()

        self.explore_decay = explore_decay
        
        self.training = training
        

    def run(self, num_episodes: int = 50000):
        """
        Runs the simulation for a given number of episodes.
        """
        A_wins = 0
        B_wins = 0
        for episode in tqdm(range(num_episodes)):

            #print("restarting episode", episode)
            self.state = self.environment.getCurrentState()
            done = False
            while not done:
                previous_state = copy.deepcopy(self.state)
                if self.environment.current_player == self.agentA.agent_index:
                    actionA = self.agentA.getAction(self.state, self.environment.getPossibleActions(self.state, self.agentA.agent_index))
                    rewardA, self.state, done = self.environment.doAction(actionA, 0)
                    if not done:
                        actionB = self.agentB.getAction(self.state, self.environment.getPossibleActions(self.state, self.agentB.agent_index))
                        rewardB, self.state, done = self.environment.doAction(actionB, 1)
                    if self.training:
                        self.agentA.updateQValue(previous_state,
                                                            self.state, 
                                                            actionA, 
                                                            actionB, 
                                                            self.environment.getPossibleActions(previous_state, self.agentA.agent_index), 
                                                            self.environment.getPossibleActions(previous_state, 1 - self.agentA.agent_index), 
                                                            rewardA)

                else:
                    actionB = self.agentB.getAction(self.state, self.environment.getPossibleActions(self.state, self.agentB.agent_index))
                    rewardB, self.state, done = self.environment.doAction(actionB, 1)
                    if not done:
                        actionA = self.agentA.getAction(self.state, self.environment.getPossibleActions(self.state, self.agentA.agent_index))
                        rewardA, self.state, done = self.environment.doAction(actionA, 0)
                   
                
                # Update Q-values    TODO: figure out if previous state should be the state before actions or the state before the actual action of the agent
                if self.training:
                    self.agentA.updateQValue(previous_state, 
                                                            next_state, 
                                                            actionA, 
                                                            actionB, 
                                                            '''