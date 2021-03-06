from collections import deque
from keras.models import kModel as kModel
from keras.layers import Input, Embedding, LSTM, Dense, Dot, GRU
from keras.layers import SimpleRNN as RNN
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from numpy.random import choice, rand
import pickle

class DDQNAgent:
    """
        Creates 2 DQN models
        Trains one on the predictions of the other
    """

    def __init__(self):
        self.EMBEDDING_SIZE = 16
        self.RNN_HIDDEN_LAYERS = 32 
        self.DENSE_LAYER = 8
        self.memory = deque(maxlen=2000)
        self.positive_memory = deque(maxlen=2000)
        self.prioritized_fraction = 0.25
        self.gamma = 0.75    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.vocab_size = 1200
        self.rnn_type = 'lstm'  # vanilla, gru, lstm
        self.exploration_strategy = 'eps' # [eps, multinomial]
        self.MAX_ACTIONS = 10**10  # updating q function requires iterating over all the actions. Cap that limit
        self.state_q_values = dict()
        self.model_histories = list()
        self.model = self.build_dqn_model_1()
        self.build_dqn_model_2()

    def build_dqn_model_1(self):
        embedding_shared = Embedding(
            self.vocab_size + 1, self.EMBEDDING_SIZE, input_length=None, mask_zero=True, trainable=True, name="embedding_shared"
        )

        if self.rnn_type == 'lstm':
            rnn_shared = LSTM(self.RNN_HIDDEN_LAYERS, name="rnn_shared")
        if self.rnn_type == 'gru':
            rnn_shared = GRU(self.RNN_HIDDEN_LAYERS, name="rnn_shared")
        if self.rnn_type == 'vanilla':
            rnn_shared = RNN(self.RNN_HIDDEN_LAYERS, name="rnn_shared")

        # create model for state
        input_state = Input(batch_shape=(None, None), name="input_state")
        embedding_state = embedding_shared(input_state)
        rnn_state = rnn_shared(embedding_state)
        dense_state = Dense(self.DENSE_LAYER , activation='linear', name="dense_state")(rnn_state)
        self.state_model_dqn_1 = kModel(inputs=input_state, outputs=dense_state, name="state")
    
        # create model for action
        input_action = Input(batch_shape=(None, None), name="input_action")
        embedding_action = embedding_shared(input_action)
        rnn_action = rnn_shared(embedding_action)
        dense_action = Dense(self.DENSE_LAYER , activation='linear', name="dense_action")(rnn_action)
        self.action_model_dqn_1 = kModel(inputs=input_action, outputs=dense_action, name="action")

        # create joint final linear layer
        input_dot_state = Input(shape=(self.DENSE_LAYER))
        input_dot_action = Input(shape=(self.DENSE_LAYER))
        dot_state_action = Dot(axes=-1, normalize=False, name="dot_state_action")([input_dot_state, input_dot_action])

        self.model_dot_state_action = kModel(
            inputs=[input_dot_state, input_dot_action], outputs=dot_state_action, name="dot_state_action"
        )

        model = kModel(
            inputs=[self.state_model_dqn_1.input, self.action_model_dqn_1.input], 
            outputs=self.model_dot_state_action([self.state_model_dqn_1.output, self.action_model_dqn_1.output])
        )
        model.compile(optimizer='RMSProp', loss='mse')
        return model

    def build_dqn_model_2(self):
        embedding_shared = Embedding(
            self.vocab_size + 1, self.EMBEDDING_SIZE, input_length=None, mask_zero=True, trainable=True, name="embedding_shared"
        )

        if self.rnn_type == 'lstm':
            rnn_shared = LSTM(self.RNN_HIDDEN_LAYERS, name="rnn_shared")
        if self.rnn_type == 'gru':
            rnn_shared = GRU(self.RNN_HIDDEN_LAYERS, name="rnn_shared")
        if self.rnn_type == 'vanilla':
            rnn_shared = RNN(self.RNN_HIDDEN_LAYERS, name="rnn_shared")

        # create model for state
        input_state = Input(batch_shape=(None, None), name="input_state")
        embedding_state = embedding_shared(input_state)
        rnn_state = rnn_shared(embedding_state)
        dense_state = Dense(self.DENSE_LAYER , activation='linear', name="dense_state")(rnn_state)
        self.state_model_dqn_2 = kModel(inputs=input_state, outputs=dense_state, name="state")
    
        # create model for action
        input_action = Input(batch_shape=(None, None), name="input_action")
        embedding_action = embedding_shared(input_action)
        rnn_action = rnn_shared(embedding_action)
        dense_action = Dense(self.DENSE_LAYER , activation='linear', name="dense_action")(rnn_action)
        self.action_model_dqn_2 = kModel(inputs=input_action, outputs=dense_action, name="action")

        # create joint final linear layer
        input_dot_state = Input(shape=(self.DENSE_LAYER))
        input_dot_action = Input(shape=(self.DENSE_LAYER))
        dot_state_action = Dot(axes=-1, normalize=False, name="dot_state_action")([input_dot_state, input_dot_action])

        self.model_dot_state_action_double = kModel(
            inputs=[input_dot_state, input_dot_action], outputs=dot_state_action, name="dot_state_action"
        )

        model = kModel(
            inputs=[self.state_model_dqn_2.input, self.action_model_dqn_2.input], 
            outputs=self.model_dot_state_action_double([self.state_model_dqn_2.output, self.action_model_dqn_2.output])
        )
        model.compile(optimizer='RMSProp', loss='mse')

        # no need to return the mo del
        #



    def save_model_weights(self):
        self.model.save('zork_model.h5')
        self.model.save_weights('zork_model_weights.h5')
        try:
            with open('zork_model_history.pickle', 'wb') as fp:
                pickle.dump(self.model_histories, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass

    def remember(self, state, _, action, reward, next_state, next_state_text, action_dict, done):
        self.memory.append((state, action, reward, next_state, next_state_text, action_dict, done))
        if reward > 0.5:
            self.positive_memory.append((state, action, reward, next_state, next_state_text, action_dict, done))

    def predict_actions(self, state_text, state, action_dict, actions):
        state_dense = self.state_model_dqn_1.predict([state])[0]
        state_input = state_dense.reshape((1, len(state_dense)))
        
        if self.exploration_strategy == 'eps':
            ## decide which type of action to perform
            if rand() <= self.epsilon:
                random_index = choice(len(actions))
                best_action = actions[random_index]
            else: 
                best_action, _ = self.compute_max_q(state_text, state_input, action_dict)
        else:
            best_action, _ = self.compute_q_multinomial(state_text, state_input, action_dict)

        return best_action

    def compute_q_multinomial(self, state_text, state_input, action_dict):
        if state_text in self.state_q_values:
            q_target = self.state_q_values[state_text]
        else:
            q_target = 0
            self.state_q_values[state_text] = q_target
        action_items = list(action_dict.items())
        N = len(action_items)

        # selecting actions using thompson sampling
        probs = np.array([action[1][0] for action in action_items])
        probs_norm = probs / probs.sum()
        
        idx = choice(list(range(N)), size=1, p=probs_norm)[0]
        action, data = action_items[idx]
        _, action_vector = data
        action_dense = self.action_model_dqn_2.predict([action_vector], use_multiprocessing=True)[0]
        action_input = action_dense.reshape((1, len(action_dense)))
        q = self.model_dot_state_action_double.predict([state_input, action_input], use_multiprocessing=True)[0][0]

        self.state_q_values[state_text] = q
        return action, q

    
    def compute_max_q(self, state_text, state_input, action_dict):
        if state_text in self.state_q_values:
            q_target = self.state_q_values[state_text]
        else:
            q_target = 0
            self.state_q_values[state_text] = q_target
        i = 0
        q_max = -1e20
        action_items = list(action_dict.items())
        N = len(action_items)
        
        for i in range(min(N, self.MAX_ACTIONS)):
            if q_max < q_target:
                action, data = action_items[i]
                _, action_vector = data
                action_dense = self.action_model_dqn_2.predict([action_vector], use_multiprocessing=True)[0]
                action_input = action_dense.reshape((1, len(action_dense)))
                q = self.model_dot_state_action_double.predict([state_input, action_input], use_multiprocessing=True)[0][0]
                if q > q_max:
                    q_max = q
                    best_action = action
            else:
                break

        self.state_q_values[state_text] = q_max
        return best_action, q_max
        
    def replay(self, batch_size):
        states = [None]*batch_size
        actions = [None]*batch_size
        targets = np.zeros((batch_size, 1))
        next_state_dict = pd.DataFrame(columns=['next_state', 'next_state_input', 'future_q'])
        batch_positive_size = int(batch_size*self.prioritized_fraction)
        batch_normal_size = batch_size - batch_positive_size
        batch_positive_selections = choice(len(self.positive_memory), batch_positive_size)
        batch_normal_selections = choice(len(self.memory), batch_normal_size)
        b_p = 0
        b_r = 0 
        for i in range(batch_size):  
            if i < batch_positive_size:  ## get positive experience
                state, action, reward, next_state, next_state_text, action_dict, done = self.positive_memory[batch_positive_selections[b_p]]
                b_p += 1
            else: 
                state, action, reward, next_state, next_state_text, action_dict, done = self.memory[batch_normal_selections[b_r]]
                b_r += 1
            target = reward
            if not done:
                try:
                    next_state_input = next_state_dict[next_state_dict['next_state'] == next_state]['next_state_input'] 
                    future_q = next_state_dict[next_state_dict['next_state'] == next_state]['future_q']
                except:
                    next_state_dense = self.state_model_dqn_2.predict([next_state])[0]
                    next_state_input = next_state_dense.reshape((1, len(next_state_dense)))

                    if self.exploration_strategy == 'eps':
                        _, future_q = self.compute_max_q(next_state_text, next_state_input, action_dict)
                    else:
                        _, future_q = self.compute_q_multinomial(next_state_text, next_state_input, action_dict)
                        
                    row = len(next_state_dict)
                    next_state_dict.loc[row, 'next_state'] = next_state
                    next_state_dict.loc[row, 'next_state_input'] = next_state_input
                    next_state_dict.loc[row, 'future_q'] = future_q

                ## calculate target
                target = reward + self.gamma*future_q
                
                ## store state, action, target
                states[i] = state[0]
                actions[i] = action[0]
                targets[i] = target
                
        history = self.model.fit(x=[
            pad_sequences(states), 
            pad_sequences(actions)
        ], y=targets, batch_size=batch_size, epochs=1, verbose=1)
        self.model_histories.append(history)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
