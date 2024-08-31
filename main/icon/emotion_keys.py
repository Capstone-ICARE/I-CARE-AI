from icon.data_load import DataLoader

class EmotionKeys:
    def __init__(self):
        self.loader = DataLoader()
        
    def predict_emotion(self, diary_content):
        vector = self.loader.get_sentence_vector(diary_content)
        predicted_key = self.loader.emotion_model.predict([vector])
        return str(predicted_key[0])

    def add_emotion_keys(self, word, matched_keys_emotion):
        emotion = self.loader.inverse_emotion_dict[word]
        if emotion not in matched_keys_emotion:
            matched_keys_emotion.append(emotion)

    def get_emotion_keys(self, diary_content):
        matched_keys_emotion = []
        morphs = self.loader.okt.morphs(diary_content, stem=True)
        converted_verbs = []
        skip_next = False

        for i in range(len(morphs)):
            if skip_next:
                skip_next = False
                continue
            if i < len(morphs) - 1:
                combined = morphs[i] + morphs[i+1]
                if combined in self.loader.inverse_emotion_dict:
                    converted_verbs.append(combined)
                    self.add_emotion_keys(combined, matched_keys_emotion)
                    skip_next = True
                    continue
            if morphs[i] in self.loader.inverse_emotion_dict:
                converted_verbs.append(morphs[i])
                self.add_emotion_keys(morphs[i], matched_keys_emotion)

        predicted_key = self.predict_emotion(diary_content)
        return matched_keys_emotion, predicted_key
    