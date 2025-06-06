# AAMAS_Markov_Games

To use run_simulation, modify the following variables in main:

A_type and B_type:
Strings indicating the type of each agent. Valid options:
"random" – takes random actions
"probabilistic_q" – Q-learning agent with probabilistic action selection
"minimax" – uses minimax Q-learning
"epsilon_greedy" - Used to train q
"q" – standard Q-learning agent
"mock" – human-controlled agent (uses keyboard input)
"handcrafted" – rule-based agent

env:
Environment type, SoccerEnvironment() or CatchEnvironment().

explore:
Initial exploration rate (epsilon) for "probabilistic_q", "minimax","epsilon_greedy", and "q" agents.
Has no effect on other types.

decay:
Learning rate decay, only applies to "probabilistic_q", "minimax", "epsilon_greedy", and "q" agents.

explore_decay:
Rate at which the exploration value decreases over time.
Also only affects "probabilistic_q", "minimax", "epsilon_greedy", and "q".

timesteps:
Integer for how many time steps the simulation should run.

training:
Boolean. Set to True if you want the agents to learn during the simulation.

use_gui:
Boolean. Set to True if you want a graphical interface.

mac:
Boolean. If using a GUI on macOS, set this to True.

extra:
List of two strings used when loading trained agents. Each string is a two-letter code.
The first letter is the type of the agent; the second is the type it was trained against.
For example, ["qr", "qq"] loads a Q-learning agent trained vs Random, and a Q-learning agent trained vs another Q-learning agent.
Some agent combinations may not have trained models available.

A working example is function test_trained().

## Challenger Evaluation
To run the evaluation against the respective challengers, start the script run_simulation_challengers. The functions to run MR, MM, QR and QQ against their challenger are already set. For running the evaluation against the JAL challenger, run the file run_simulation_jal.
