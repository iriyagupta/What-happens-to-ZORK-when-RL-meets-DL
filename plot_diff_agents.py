import matplotlib.pyplot as plt
import pandas as pd

dqn_multinomial_scores = pd.read_csv('game_scores_dqn_multi.csv')
random_scores = pd.read_csv('game_scores_random.csv')
dqn_max_scores = pd.read_csv('game_scores_dqn_max.csv')

# plot lines
plt.plot(dqn_multinomial_scores['Game Number'], dqn_multinomial_scores['Score'], label = "DQN (multinomial)")
plt.plot(dqn_max_scores['Game Number'], dqn_max_scores['Score'], label = "DQN (max)")
plt.plot(random_scores['Game Number'], random_scores['Score'], label = "Random Agent")
plt.xlabel('Game')
plt.ylabel('Score')
plt.title('End Game Scores')
plt.legend()
plt.show()