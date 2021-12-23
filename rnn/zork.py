from game import text_game

game = text_game()
a = game.agent

game.run_game(agent=a, num_games=100, num_rounds=100, batch_size=64, training=True)
game_scores8 = game.end_game_scores.copy()
game_scores8.to_csv('game_score_rnn8.csv')