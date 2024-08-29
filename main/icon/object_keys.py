from icon.data_load import DataLoader
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

class ObjectKeys:
    def __init__(self):
        self.loader = DataLoader()
        
    def get_combined_vectors_for_prediction(self, word, sentence):
        word_vector = self.loader.get_word_vectors(word, word)[0].mean(dim=0).detach().numpy()
        sentence_vectors = [vector.mean(dim=0).detach().numpy() for vector in self.loader.get_word_vectors(sentence, word)]
        combined_vectors = [np.concatenate((word_vector, vector)) for vector in sentence_vectors]
        return combined_vectors

    def verify_object(self, word, sentence):
        if word not in sentence:
            return 0
        combined_vectors = self.get_combined_vectors_for_prediction(word, sentence)
        predicted_label = 0
        for vector in combined_vectors:
            predicted_label = self.loader.object_model.predict([vector])[0]
            if predicted_label != 0:
                break
        return predicted_label

    def get_all_related_words(self, word):
        related_words = []
        related_words.append(word)
        if word in self.loader.word_object_synonym:
            for synonym in self.loader.word_object_synonym[word]:
                related_words.append(synonym)
        return related_words

    def contains_word(self, sentence, word):
        pattern = re.compile(re.escape(word.replace(" ", "")), re.IGNORECASE)
        sentence_replace = sentence.replace(" ", "")
        return bool(pattern.search(sentence_replace))

    def sort_important_nouns(self, diary_content):
        newstr = re.sub(r'[^ㄱ-ㅣ가-힣]+', ' ', diary_content)
        sentence_nouns = self.loader.okt.nouns(newstr)
        sentence_vector = np.mean([self.loader.fasttext_model.get_word_vector(noun) for noun in sentence_nouns], axis=0)
        distances = {noun: cosine_similarity([self.loader.fasttext_model.get_word_vector(noun)], [sentence_vector])[0][0] for noun in sentence_nouns}
        return sorted(distances.items(), key=lambda item: item[1], reverse=True)[:5]

    def append_keys_object(self, sorted_nouns, word, key, first_keys_object, matched_keys_object):
        if word in sorted_nouns:
            first_keys_object.append(key)
        else:
            matched_keys_object.append(key)

    def predict_keys_object(self, sorted_nouns, diary_content):
        most_important_noun = sorted_nouns[0][0]
        most_important_vector = self.loader.fasttext_model.get_word_vector(most_important_noun)
  
        similarities = {}
        for key, word in self.loader.match_object.items():
            word_vector = self.loader.fasttext_model.get_word_vector(word)
            similarity = cosine_similarity([most_important_vector], [word_vector])[0][0]
            if similarity > 0.34:
                if word in self.loader.trained_words:
                    if word in diary_content:
                        if self.verify_object(word, diary_content) == 1:
                            similarities[key] = similarity
                else:
                    similarities[key] = similarity
  
        predicted_keys = [key for key, _ in sorted(similarities.items(), key=lambda item: item[1], reverse=True)]
        return predicted_keys[:5]

    def get_object_keys(self, diary_content):
        first_keys_object = []
        matched_keys_object = []
        predicted_keys_object = []

        sorted_nouns = self.sort_important_nouns(diary_content)

        for key, word in self.loader.match_object.items():
            if key not in matched_keys_object:
                related_words = self.get_all_related_words(word)
                for related_word in related_words:
                    if self.contains_word(diary_content, related_word):
                        diary_content = diary_content.replace(related_word, word)
                        break
                if word in diary_content:
                    if word not in self.loader.trained_words:
                        self.append_keys_object(sorted_nouns, word, key, first_keys_object, matched_keys_object)
                    else:
                        verify = self.verify_object(word, diary_content)
                        if verify == 0:
                            if word in sorted_nouns:
                                del sorted_nouns[word]
                            else:
                                if word in self.loader.duplicate_object_keys:
                                    key = self.loader.duplicate_object_keys[word][verify - 1]
                                    self.append_keys_object(sorted_nouns, word, key, first_keys_object, matched_keys_object)
                                else:
                                    self.append_keys_object(sorted_nouns, word, key, first_keys_object, matched_keys_object)
        if (len(first_keys_object) + len(matched_keys_object)) < 2:
            predicted_keys_object.extend(self.predict_keys_object(sorted_nouns, diary_content))
        return first_keys_object, matched_keys_object, predicted_keys_object
    