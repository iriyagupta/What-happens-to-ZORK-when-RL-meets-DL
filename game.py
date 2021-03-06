from spacy.matcher import Matcher
from spacy.attrs import POS
from nltk.tokenize import word_tokenize
from itertools import permutations
from keras.preprocessing.sequence import pad_sequences
import pickle
from scipy import spatial
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

from agent import DDQNAgent
import spacy
import re
import textworld

# Let the environment know what information we want as part of the game state.
infos = textworld.EnvInfos(
    feedback=True,    # Response from the game after typing a text command.
    description=True, # Text describing the room the player is currently in.
    inventory=True    # Text describing the player's inventory.
)

VOCAB_SIZE = 1200
STATE_LIMIT = 1000

negative_per_turn_reward = 1
inventory_reward_value = 3
new_area_reward_value = 2
moving_around_reward_value = 0.5
inventory_not_new_reward_value = 0.5

class PlayZork:
    def __init__(self):
        self.agent = DDQNAgent()
        self.batch_size = None
        self.save_data = True
        
        # We are now ready to start the game.
        self.env = textworld.start('zork1.z5', infos=infos)
        self.word_2_vec = self.init_word2vec()
        
        self.nlp = spacy.load('en_core_web_sm')
        
        self.tokenizer = None
        
        self.random_action_weight = 6
        self.random_action_basic_prob = 0.4
        self.random_action_low_prob = 0.1
        
        self.score = 0
        self.game_score = 0
        self.game_score_weight = 1
                
        self.basic_actions = ['go north', 'go south', 'go west', 'go east', 'go northeast', 'go northwest', 'go southeast', 'go southwest', 'go down', 'go up']
        self.action_space = set(self.basic_actions)
        self.directions = ['north', 'south', 'east', 'west', 'northwest', 'northeast', 'southwest', 'southeast', 'up', 'down']
        self.command1_actions = ['open OBJ', 'get OBJ', 'eat OBJ', 'ask OBJ', 'make OBJ', 'wear OBJ', 'move OBJ', 'kick OBJ', 'find OBJ', 'play OBJ', 'feel OBJ', 'read OBJ', 'fill OBJ', 'pick OBJ', 'pour OBJ', 'pull OBJ', 'leave OBJ', 'break OBJ', 'enter OBJ', 'shake OBJ', 'banish OBJ', 'read OBJ', 'enchant OBJ', 'feel OBJ', 'pour OBJ']
        self.command2_actions = ['pour OBJ on DCT', 'hide OBJ in DCT', 'pour OBJ in DCT', 'move OBJ in DCT', 'hide OBJ on DCT', 'flip OBJ for DCT', 'fix OBJ with DCT', 'spray OBJ on DCT', 'dig OBJ with DCT', 'cut OBJ with DCT', 'pick OBJ with DCT', 'pour OBJ from DCT', 'fill OBJ with DCT', 'burn OBJ with DCT', 'flip OBJ with DCT', 'read OBJ with DCT', 'hide OBJ under DCT', 'carry OBJ from DCT', 'inflate OBJ with DCT', 'unlock OBJ with DCT', 'give OBJ to DCT', 'carry OBJ to DCT', 'spray OBJ with DCT']
        self.filtered_tokens = ['Score', 'Moves']
        self.invalid_nouns = [] 
        self.valid_nouns = []
        
        self.unique_state = set()
        self.actions_probs_dict = dict()
        self.story = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Reward', 'Reward_Type', 'Score', 'Moves', 'Total_Moves'])
        self.end_game_scores = pd.DataFrame(columns=['Game Number', 'Score'])
        self.stories = []
        
        self.load_invalid_nouns()
        self.load_valid_nouns()
        self.load_tokenizer()
        
        self.unique_inventory_changes = set()
        self.state_data = pd.DataFrame(columns=['State', 'StateVector', 'ActionData', 'Nouns'])
        
    def load_state_data(self):
        try:
            self.state_data = pd.read_pickle('state_data.pickle')
        except:
            self.state_data = pd.DataFrame(columns=['State', 'StateVector', 'ActionData', 'Nouns'])

    def start_game(self):
        self.load_state_data()
        self.story = pd.DataFrame(columns=['Surroundings', 'Inventory', 'Action', 'Response', 'Reward', 'Reward_Type', 'Score', 'Moves', 'Total_Moves'])
        self.end_game_scores = pd.DataFrame(columns=['Game Number', 'Score'])
        self.score = 0
        self.unique_state = set()
        self.game_score = 0
        self.game_state = self.env.reset()

    def kill_game(self):
        if self.save_data:
            self.save_invalid_nouns()
            self.save_valid_nouns()
            self.save_tokenizer()
            self.agent.save_model_weights()
        
    def restart_game(self):
        self.save_invalid_nouns()
        self.save_valid_nouns()
        self.env.step('restart')
        self.env.step('y') # confirm
        self.score = 0
        self.unique_state = set()
        self.unique_inventory_changes = set()
        self.game_score = 0
        
    def save_model_weights(self):
        self.agent.model.save_weights('ddqn_model_weights.h5')
        
    def getGameState(self):
        description = self.game_state.description
        description = description.replace('\n', " ").replace('\r', " ")

        # 840726 marks the start of the game
        if '840726' in description:
            description = description[description.index('840726') + len('840726'):]
            
        return description
        
    def get_state(self):
        surroundings = self.preprocess(self.game_state.description)
        inventory = self.preprocess(self.game_state.inventory)
        score = self.game_state.score
        state = surroundings + ' ' + inventory
        return state, inventory, score

    def get_nouns(self, state):
        matcher = Matcher(self.nlp.vocab)
        matcher.add('Noun phrase', None, [{POS: 'NOUN'}])
        doc = self.nlp(state)
        matches = matcher(doc)
        noun_set = set()
        for id, start, end in matches:
            noun = doc[start:end].text
            if noun not in self.directions and noun not in self.invalid_nouns:
                noun_set.add(noun)
        return noun_set
        
    def generate_action_tuples(self, nouns):
        possible_actions = []
        similarities = []
        for i in self.basic_actions:
            possible_actions.append(i)
            similarities.append(self.random_action_basic_prob)
        for i in nouns:
            for action1 in self.command1_actions:   
                action_to_add = action1.replace('OBJ', i)
                possible_actions.append(action_to_add)
                try:
                    similarities.append(self.word_2_vec.similarity(word_tokenize(action_to_add)[0], i))
                except:
                    similarities.append(self.random_action_low_prob)
            noun_permutations = list(permutations(nouns, 2))    
            for action2 in self.command2_actions:
                for perm in noun_permutations:
                    if (perm[0] == perm[1]):  
                        pass
                    else:
                        action_to_add = action2.replace('OBJ', perm[0])
                        action_to_add = action_to_add.replace('DCT', perm[1])
                        possible_actions.append(action_to_add)
                        try:
                            similarities.append(self.word_2_vec.similarity(word_tokenize(action_to_add.strip(i))[0], i).mean())
                        except:
                            similarities.append(self.random_action_low_prob)

        return possible_actions

    def embedding_similarity(self, verb, noun, agent):
        embedding_matrix = agent.state_model_dqn_1.layers[1]
        embedding_matrix = embedding_matrix.get_weights()
        embedding_matrix = embedding_matrix[0]
        noun_idx, verb_idx = self.tokenizer.texts_to_sequences(word_tokenize(noun)), self.tokenizer.texts_to_sequences(word_tokenize(verb))
        noun_vec, verb_vec = embedding_matrix[noun_idx[0]], embedding_matrix[verb_idx[0]]
        return 1 - spatial.distance.cosine(noun_vec, verb_vec)
    
    def add_to_action_space(self, action_space, actions):
        similarities = []

        for action in actions:
            action_space.add(action)
        for action in action_space:
            words = word_tokenize(action)
            verb = words[0]
            if verb in self.basic_actions:    
                similarities.append(self.random_action_basic_prob)
            elif len(words)<3:           
                noun = word_tokenize(action)[1]
                # try:
                if self.agent is None:
                    sim_score = self.word_2_vec.similarity(verb, noun)**self.random_action_weight
                else:
                    sim_score = np.abs(self.embedding_similarity(verb, noun, self.agent))**self.random_action_weight
                if sim_score < 0:
                    sim_score = self.random_action_basic_prob**self.random_action_weight
                similarities.append(sim_score)
            else:                       
                # try:
                noun1 = word_tokenize(action)[1]
                prep = word_tokenize(action)[2]
                noun2 = word_tokenize(action)[3]
                if self.agent is None:
                    sim_score1 = self.word_2_vec.similarity(verb, noun1)
                    sim_score2 = self.word_2_vec.similarity(prep, noun2)
                else:
                    sim_score1 = np.abs(self.embedding_similarity(verb, noun1, self.agent))
                    sim_score2 = np.abs(self.embedding_similarity(prep, noun2, self.agent))
                sim_score = ((sim_score1 + sim_score2)/2)**self.random_action_weight
                if sim_score < 0:
                    sim_score = 0.05
                similarities.append(sim_score**self.random_action_weight)

        return action_space, similarities
        
    def perform_action(self, command):
        self.game_state, self.reward, self.done = self.env.step(command)
        
    def preprocess(self, text):
        # fix bad newlines (replace with spaces), unify quotes
        text = text.strip()
        text = text.replace('\\n', '').replace('???', '\'').replace('???', '\'').replace('???', '"').replace('???', '"')
        # convert to lowercase
        text = text.lower()
        # remove all characters except alphanum, spaces and - ' "
        text = re.compile('[^ \-\sA-Za-z0-9"\']+').sub('', text)
        text = re.sub('\s{2,}', ' ', text)
        return text

    def vectorize_text(self, text, tokenizer):
        words = word_tokenize(text)
        tokenizer.fit_on_texts(words)
        seq = tokenizer.texts_to_sequences(words)
        sent = []
        for i in seq:
            sent.append(i[0])
        padded = pad_sequences([sent], maxlen=50, padding='post')
        return (padded)
    
    def calculate_reward(self, inventory, old_inventory, moves_count, old_state, new_state, round_score):
        reward = 0
        reward_msg = ''
        
        if(moves_count != 0):
            reward = reward + round_score*self.game_score_weight
            if (round_score > 0):
                print('Scored ' + str(round_score) + ' points in game.')
                reward_msg += ' game score: ' + str(round_score) + ' '
        
        reward = reward - negative_per_turn_reward
        
        if(moves_count != 0):
            if inventory.strip().lower() not in old_inventory.strip().lower(): 
                
                if (old_inventory + ' - ' + inventory) not in self.unique_inventory_changes:
                    self.unique_inventory_changes.add(old_inventory + ' - ' + inventory)
                    reward = reward + inventory_reward_value
                    print('inventory changed - new')
                    reward_msg += ' inventory score (' + old_inventory + " --- " + inventory + ')'
                else:
                    reward = reward + inventory_not_new_reward_value
                    
        
        if new_state.strip() not in self.unique_state:  
            reward = reward + new_area_reward_value
            self.unique_state.add(new_state.strip())
            reward_msg += ' new area score ---' + new_state.strip()
        
        if old_state not in new_state:
            reward = reward + moving_around_reward_value
            reward_msg += ' - moved around - ' 

        
        # print('Rewarded: ' + str(reward) + ' points.')
        return reward, reward_msg

    def detect_invalid_nouns(self, action_response):
        word = ''
        
        if('know the word' in action_response):
            startIndex = action_response.find('\"')
            endIndex = action_response.find('\"', startIndex + 1)
            word = action_response[startIndex+1:endIndex]
        return word
            
    def save_tokenizer(self):
        with open('tokenizer.pickle', 'wb') as fp:
            pickle.dump(self.tokenizer, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_tokenizer(self):
        with open('tokenizer.pickle', 'rb') as fp:
            self.tokenizer = pickle.load(fp)
       
    def save_invalid_nouns(self):
        with open('invalid_nouns.pickle', 'wb') as fp:
            pickle.dump(self.invalid_nouns, fp)
    
    def load_invalid_nouns(self):
        with open ('invalid_nouns.pickle', 'rb') as fp:
            lst = pickle.load(fp)
            self.invalid_nouns.extend(lst)
        
    def save_valid_nouns(self):
        with open('valid_nouns.pickle', 'wb') as fp:
            pickle.dump(self.valid_nouns, fp)
    
    def load_valid_nouns(self):
        with open ('invalid_nouns.pickle', 'rb') as fp:
            lst = pickle.load(fp)
            self.valid_nouns.extend(lst)
    
    def init_word2vec(self):
        # preload word2vec model
        with open('demodock.txt', 'r', encoding="ISO-8859-1") as f:
            tutorials = f.read()
            sentences = word_tokenize(tutorials)
            w2v = gensim.models.Word2Vec([sentences])
            return w2v
        
    def get_data(self, state):
        
        if (state in list(self.state_data['State'])):
            state_vector = list(self.state_data[self.state_data['State'] == state]['StateVector'])[0][0]
            try:
                nouns = list(self.state_data[self.state_data['State'] == state]['Nouns'])[0][0]
            except:
                None
            try:
                actionsVectors = []
                actions = []
                probs = []
                action_dict = list(self.state_data[self.state_data['State'] == state]['ActionData'])[0]
                for act, data in action_dict.items():
                    actions.append(act)
                    probs.append(data[0])
                    actionsVectors.append(data[1])
                probs = np.array(probs)
            except:
                actionsVectors = []
                actions = []
                probs = []
                action_dict = list(self.state_data[self.state_data['State'] == state]['ActionData'])[0][0]
                for act, data in action_dict.items():
                    actions.append(act)
                    probs.append(data[0])
                    actionsVectors.append(data[1])
                probs = np.array(probs)
        else: 
            state_vector = self.vectorize_text(state,self.tokenizer)
            
            nouns = self.get_nouns(state)
            
            self.test_nouns(nouns)
            
            for noun in nouns:
                if noun in self.invalid_nouns:
                    nouns.remove(noun)
            # build action space and probabilities 
            current_action_space = self.generate_action_tuples(nouns)
            action_space = set()
            action_space, probs = self.add_to_action_space(action_space, current_action_space)
            actions = []
            for a in action_space:
                actions.append(a)
            probs = np.array(probs)
            actionsVectors = []
            for a in actions:
                actionsVectors.append(self.vectorize_text(a,self.tokenizer))
            
            action_dict = dict()
            for idx, act in enumerate(actions):
                action_dict[act] = (probs[idx], actionsVectors[idx])
            
            row = len(self.state_data)
            self.state_data.loc[row, 'State'] = state
            self.state_data.loc[row, 'StateVector'] = [state_vector]
            self.state_data.loc[row, 'ActionData'] = [action_dict]
            self.state_data.loc[row, 'Nouns'] = [nouns]
        return probs, actions, state_vector, actionsVectors, action_dict
    
    def perform_selected_action(self, action):
        self.perform_action(action)
        response = self.getGameState()
        response = self.preprocess(response)
        return response
    
    def test_nouns(self, nouns):
        for noun in nouns:
            if noun in self.invalid_nouns or noun in self.valid_nouns:
                pass
            else:
                action = 'feel ' + noun
                response, _, _ = self.env.step(action)
                if('know the word' in response):
                    self.invalid_nouns.append(noun)
                else:
                    self.valid_nouns.append(noun)
                    
    def detect_invalid_action(self, state, action, reward, action_dict, invalid_noun):
        
        invalid_action = ''
        if (reward==-1):
            invalid_action = action
        
        if invalid_noun:
            for act, _ in action_dict.items():
                if invalid_noun in act:
                    del action_dict[invalid_noun]
            self.state_data.loc[self.state_data['State'] == state, 'ActionData'] = [action_dict]
        if invalid_action and invalid_action in action_dict:
            del action_dict[invalid_action]
            
            self.state_data.loc[self.state_data['State'] == state, 'ActionData'] = [action_dict]
        return action_dict
                      
                    
    def run_game(self, num_games, num_rounds, batch, training):
        self.batch_size = batch
        self.start_game()

        for game_number in tqdm(range(num_games)):
            new_state = ''
            inventory = ''
            
            for i in range(num_rounds):
                
                if (i==0):
                    state, old_inventory, _ = self.get_state()
                    self.unique_state.add(state)
                else:
                    state  = new_state
                    old_inventory = inventory
                    
                invalid_line = True
                while invalid_line:
                    invalid_line = False
                    if len(state) > STATE_LIMIT or len(state)<5 or 'score' in state:
                        print('encountered line read bug')
                        state, old_inventory, _ = self.get_state()
                        invalid_line = True
                
                _, actions, state_vector, _, action_dict = self.get_data(state)
            
                # find action using selected exploration strategy
                action = self.agent.predict_actions(state, state_vector, action_dict, actions)

                response = self.perform_selected_action(action)
                
                invalid_noun = self.detect_invalid_nouns(response)
                
                action_vector = self.vectorize_text(action,self.tokenizer)

                new_state, inventory, current_score = self.get_state()
                new_state = self.preprocess(new_state)
                new_state_vector = self.vectorize_text(new_state, self.tokenizer)
                
                round_score = current_score - self.game_score
                self.game_score = current_score
                reward, reward_msg = self.calculate_reward(inventory, old_inventory, i, state, new_state, round_score)

                self.score += reward
                total_round_number = i + game_number*num_rounds
                self.story.loc[total_round_number] = [state, old_inventory, action, response, reward, 
                                reward_msg, self.score, str(i), total_round_number]
                
                action_dict = self.detect_invalid_action(state, action, reward, action_dict, invalid_noun)
                _, _, new_state_vector, _, _ = self.get_data(new_state)
                if training:
                    self.agent.remember(state_vector, state, action_vector, reward, new_state_vector,
                                    new_state, action_dict, False)
                
                if training and (i+1)%self.batch_size == 0 and self.agent.positive_memory:  
                    self.agent.replay(self.batch_size)
            # ----- for number of rounds ends -----

            if self.agent.exploration_strategy == 'eps':
                print(f'Score in game {game_number}: {self.score:.2f}  Epsilon: {self.agent.epsilon:.2f}')
            else:
                print(f'Score in game {game_number}: {self.score:.2f}')
                
            self.end_game_scores.loc[game_number] = [game_number, self.score]
            self.restart_game()
            self.stories.append(self.story)

        # ----- for number of games ends -----


        if self.save_data:
            self.state_data.to_pickle('state_data.pickle')

        self.kill_game()
