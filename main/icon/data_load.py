import pandas as pd
import json
from collections import defaultdict, Counter
#from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import BertModel, BertTokenizer
import fasttext
from konlpy.tag import Okt
import pickle
import torch
import numpy as np

class DataLoader:
    def __init__(self):
        path = './icon/data/'
        self.icons = pd.read_csv(path + 'newIcon.csv')
        with open(path + 'match_object.json', 'r', encoding='utf-8') as f:
            self.match_object = json.load(f)
        with open(path + 'match_emotion.json', 'r', encoding='utf-8') as f:
            self.match_emotion = json.load(f)
        self.word_emotion_synonym = {
            '1': ['재밌다', '신나다', '즐겁다', '기쁘다', '행복하다', '소중하다', '고맙다', '뿌듯하다', '자랑스럽다', '감격하다', '만족하다', '흐뭇하다', '웃다', '맛있다', '안심하다', '괜찮다', '다짐하다'],
            '2': ['사랑하다', '좋다', '기대되다', '설레다', '반하다'],
            '3': ['놀라다', '신기하다'],
            '4': ['긴장하다', '겁나다', '무섭다', '두렵다', '걱정하다', '괴롭다'],
            '5': ['실망하다', '답답하다', '후회하다', '서운하다', '싫다'],
            '6': ['화나다', '짜증난다', '억울하다', '분하다', '어이없다', '충격받다'],
            '7': ['슬프다', '외롭다', '울다'],
            '8': ['피곤하다', '지루하다', '졸리다', '지치다', '힘들다'],
            '9': ['민망하다', '창피하다', '어색하다'],
            '10': ['아쉽다', '부럽다'],
            '11': ['아프다', '힘들다']
            }
        self.word_object_synonym = {
            '개': ['강아지', '멍멍이'],
            '용': ['드래곤'],
            '스파게티': ['파스타'],
            '책': ['소설', '이야기']
            }
        self.inverse_emotion_dict = {word: key for key, words in self.word_emotion_synonym.items() for word in words}
        self.duplicate_object_keys = self.get_duplicate_object_keys()
        
        #self.tokenizer = get_tokenizer()
        #self.bert_model = get_kobert_model()
        self.bert_model = BertModel.from_pretrained(path + "kobert")
        self.tokenizer = BertTokenizer.from_pretrained(path + "kobert")
        self.fasttext_model = self.load_fasttext_model(path)
        self.okt = Okt()
        
        with open(path + 'emotion_model.pkl', 'rb') as file:
            self.emotion_model = pickle.load(file)
        with open(path + 'object_model.pkl', 'rb') as file:
            self.object_model = pickle.load(file)
            
        training_object = pd.read_csv(path + 'sentence_object.csv', encoding='cp949')
        self.trained_words = set(training_object['word'])
    
    def get_duplicate_object_keys(self):
        object_word_counts = Counter(self.match_object.values())
        duplicate_object_words = [word for word, count in object_word_counts.items() if count > 1]
        duplicate_object_keys = defaultdict(list)
        for key, word in self.match_object.items():
            if word in duplicate_object_words:
                duplicate_object_keys[word].append(key)
        return duplicate_object_keys
    
    def load_fasttext_model(self, path):
        model_path = path + 'cc.ko.300.bin'
        #model_path = "https://drive.google.com/file/d/1O_qf_ACQNmFBcVc3gXXAuvx0QH8mRiNh/view?usp=drive_link"
        return fasttext.load_model(model_path)
        #model = fasttext.train_unsupervised(input=model_path, model='skipgram')
        #model.quantize(input=path, retrain=True)
        #return model
    
    # 문장의 문맥 벡터 추출
    def get_sentence_vector(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        cls_vector = last_hidden_states[:, 0, :]
        return cls_vector.squeeze().numpy()
    
    # 타겟 단어의 문맥 벡터 추출
    def get_word_vectors(self, sentence, target_word):
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
        token_ids = inputs['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        indices = []
        split_tokens = []
        index_map = []
        target_len = len(target_word)

        for idx, item in enumerate(tokens):
            for char in item:
                split_tokens.append(char)
                index_map.append(idx)

        for i in range(len(split_tokens) - target_len + 1):
            if split_tokens[i:i + target_len] == list(target_word):
                indices.append(list(set(index_map[i:i + target_len])))

        if not indices:  # 타겟 단어가 문장에 없을 경우
            raise ValueError(f"'{target_word}' not found in the given sentence.")

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        vectors = [torch.stack([last_hidden_states[0, idx, :] for idx in idx_list]) for idx_list in indices ]
        return vectors
    
    # word와 sentence를 결합하여 임베딩 생성
    def get_combined_vector(self, word, sentence):
        word_vector = self.get_word_vectors(word, word)[0].mean(dim=0).detach().numpy()
        sentence_vector = self.get_word_vectors(sentence, word)[0].mean(dim=0).detach().numpy()
        combined_vector = np.concatenate((word_vector, sentence_vector))
        return combined_vector
    