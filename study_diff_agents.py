from game import PlayZork

def run_ddqn():
    game = PlayZork()
    game.run_game(num_games=32, num_rounds=64, batch_size=8, training=True)

    game_scores = game.end_game_scores.copy()
    game_scores.to_csv('game_scores_ddqn.csv')


def run_random():
    game = PlayZork()
    game.agent.epsilon_decay = 1.0 # no decay at all
    game.run_game(num_games=32, num_rounds=64, batch_size=8, training=False)

    game_scores = game.end_game_scores.copy()
    game_scores.to_csv('game_scores_random.csv')


# smaller batch_size = frequent calls to replay = quick decay of epsilon
run_random()
run_ddqn()