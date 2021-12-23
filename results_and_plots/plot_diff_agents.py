import matplotlib.pyplot as plt
import pandas as pd

ddqn_multinomial_scores = pd.read_csv('game_scores_ddqn_multi.csv')
random_scores = pd.read_csv('game_scores_random.csv')
ddqn_max_scores = pd.read_csv('game_scores_ddqn_max.csv')

# plot lines
plt.plot(ddqn_multinomial_scores['Game Number'], ddqn_multinomial_scores['Score'], label = "DDQN (multinomial)")
plt.plot(ddqn_max_scores['Game Number'], ddqn_max_scores['Score'], label = "DDQN (max)")
plt.plot(random_scores['Game Number'], random_scores['Score'], label = "Random Agent")
plt.xlabel('Game')
plt.ylabel('Score')
plt.title('End Game Scores')
plt.legend()
plt.show()