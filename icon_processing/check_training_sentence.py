import pandas as pd
import json
from collections import defaultdict

with open('match_object.json', 'r', encoding='utf-8') as f:
    match_object = json.load(f)
    
training_object = pd.read_csv('sentence_object.csv', encoding='cp949')

def get_word_in_object():
  word_in_object = defaultdict(list)
  for key, word in match_object.items():
    for key, obj in match_object.items():
      if word in obj and word != obj:
        word_in_object[word].append(obj)
  return word_in_object

def check_training_object(training_object):
  no_object = defaultdict(list)
  word_in_object = get_word_in_object()
  for word, objs in word_in_object.items():
    if word in training_object['word'].unique():
      check_sentence = training_object[training_object['word'] == word]['sentence'].tolist()
      check_sentence = ' '.join(check_sentence)
      for obj in objs:
        if obj not in check_sentence:
          no_object[word].append(obj)
    else:
      for obj in objs:
        no_object[word].append(obj)
  for word, objs in no_object.items():
    print(word, ": ", objs)
  return no_object

no_object = check_training_object(training_object)