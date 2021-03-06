from game import PlayZork

def run_ddqn(epsilon):
    game = PlayZork()
    game.agent.epsilon = epsilon
    game.run_game(num_games=16, num_rounds=32, batch_size=8, training=True)

    game_scores = game.end_game_scores.copy()
    game_scores.to_csv(f'game_scores_ddqn_{epsilon}.csv')


# smaller batch_size = frequent calls to replay = quick decay of epsilon
for epsilon in [1, 0.8, 0.6, 0.4]:
    run_ddqn(epsilon)