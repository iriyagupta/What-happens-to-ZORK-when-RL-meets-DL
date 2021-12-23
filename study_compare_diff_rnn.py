from game import text_game


def train_gru_model():
    game = text_game()
    game.agent.rnn_type = 'gru'
    
    game.run_game(num_games=100, num_rounds=100, batch_size=64, training=True)
    game_scores = game.end_game_scores.copy()
    game_scores.to_csv('game_scores_dqn_gru.csv')


def train_rnn_model():
    game = text_game()
    game.agent.rnn_type = 'vanilla'

    game.run_game(num_games=100, num_rounds=100, batch_size=64, training=True)
    game_scores = game.end_game_scores.copy()
    game_scores.to_csv('game_scores_dqn_rnn.csv')

train_gru_model()
train_rnn_model()
