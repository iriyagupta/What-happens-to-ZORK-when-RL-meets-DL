# What-happens-to-ZORK-when-RL-meets-DL

This is our final project submission for the ELENE6885 Reinforcement learnig taught by Prof. Li and Prof. Wang at Columbia University

## Abstract 

Text games or popularly known as Interactive Fiction Games have been quite popular for a very long time. 
Current reinforcement learning research trends includes exploring and improving these text games which comes with lots of challenges like partial-observability and large action-state space. In this project, we explore the idea of playing text-based games by training a Reinforcement Learning (RL) agent in the textworld environment. We perform our experiments on the game Zork, which is a fantasy based text game. We investigate Deep Reinforcement Learning, with the help of different RL algorithms, such as SARSA and Q-Learning along with different RNN architectures. We also try to draw a comparison between the different agents and their performance. We show that the DDQN agent performs better than an agent that has picked an action randomly. 
Additionally, we present a brief finding on different values of $$\epsilon$$ and the effect on our game scores when the agent discovers new items. 

To install the requirements:
`pip -r requirements.txt`

To run the code, run the following files:
`study_compare_diff_rnn.py`
`study_diff_agents.py`
`study_effect_of_epsilon.py`
`study_time_action_space_size.py`
