from game import text_game

def run_ddqn():
    game = text_game()
    game.agent.epsilon = 0.8
    game.run_game(num_games=32, num_rounds=16, batch_size=4, training=True)

    game_scores = game.end_game_scores.copy()
    game_scores.to_csv(f'game_scores_ddqn_time_action.csv')


run_ddqn()