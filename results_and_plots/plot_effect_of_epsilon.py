import matplotlib.pyplot as plt
import pandas as pd

for epsilon in [1, 0.8, 0.6, 0.4]:
    scores = pd.read_csv(f'game_scores_ddqn_{epsilon}.csv')
    plt.plot(scores['Game Number'], scores['Score'], label = f"eps {epsilon}")

plt.xlabel('Game')
plt.ylabel('Score')
plt.title('DDQN Agent - End Game Scores')
plt.legend()
plt.show()