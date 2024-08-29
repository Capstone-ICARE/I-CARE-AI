from icon.data_load import DataLoader
from icon.emotion_keys import EmotionKeys
from icon.object_keys import ObjectKeys
import random

class Icon:
    def __init__(self):
        self.loader = DataLoader()
        self.emotion = EmotionKeys()
        self.object = ObjectKeys()
    
    def get_result_keys(self, diary_content):
        result_keys = []
        matched_keys_emotion, predicted_key_emotion = self.emotion.get_emotion_keys(diary_content)
        first_keys_object, matched_keys_object, predicted_keys_object = self.object.get_object_keys(diary_content)

        if matched_keys_emotion is None:
            matched_keys_emotion = []
        if first_keys_object is None:
            first_keys_object = []
        if matched_keys_object is None:
            matched_keys_object = []
        if predicted_keys_object is None:
            predicted_keys_object = []

        ln_emotion = len(matched_keys_emotion)
        ln_first_object = len(first_keys_object)
        ln_object = len(matched_keys_object)

        if (ln_first_object + ln_object) >= 2:
            if ln_first_object >= 2:
                result_keys.extend([first_keys_object[0], first_keys_object[1]])
            elif ln_first_object == 1:
                object_key = random.sample(matched_keys_object, 1)
                result_keys.extend([first_keys_object[0], object_key[0]])
            else:
                object_keys = random.sample(matched_keys_object, 2)
                result_keys.extend([object_keys[0], object_keys[1]])
        elif (ln_first_object + ln_object) == 1:
            obj1 = first_keys_object[0] if first_keys_object else matched_keys_object[0]
            result_keys.append(obj1)
            if predicted_keys_object:
                obj2 = predicted_keys_object[0]
                if obj1 != obj2:
                    result_keys.append(predicted_keys_object[0])
        else:
            if predicted_keys_object:
                result_keys.extend(predicted_keys_object[:2])

        if len(result_keys) == 2:
            if (ln_emotion == 0) or (predicted_key_emotion in matched_keys_emotion):
                result_keys.append(predicted_key_emotion)
            else:
                result_keys.append(matched_keys_emotion[-1])
        elif len(result_keys) == 1:
            if ln_emotion == 0:
                result_keys.append(predicted_key_emotion)
                result_keys.append(predicted_key_emotion)
            elif ln_emotion == 1:
                result_keys.append(predicted_key_emotion)
                result_keys.append(matched_keys_emotion[0])
            else:
                if predicted_key_emotion in matched_keys_emotion:
                    result_keys.append(predicted_key_emotion)
                    if matched_keys_emotion[-1] != predicted_key_emotion:
                        result_keys.append(matched_keys_emotion[-1])
                    else:
                        result_keys.append(matched_keys_emotion[0])
                else:
                    result_keys.append(matched_keys_emotion[-1])
                    result_keys.append(matched_keys_emotion[0])
        elif len(result_keys) == 0:
            if ln_emotion == 0:
                result_keys.extend([predicted_key_emotion] * 3)
            elif ln_emotion == 1:
                result_keys.extend([matched_keys_emotion[0], matched_keys_emotion[0], predicted_key_emotion])
            elif ln_emotion == 2:
                result_keys.extend([matched_keys_emotion[0], matched_keys_emotion[1], predicted_key_emotion])
            else:
                if predicted_key_emotion in matched_keys_emotion:
                    result_keys.append(predicted_key_emotion)
                    if matched_keys_emotion[-1] != predicted_key_emotion:
                        result_keys.append(matched_keys_emotion[-1])
                    if matched_keys_emotion[0] != predicted_key_emotion:
                        result_keys.append(matched_keys_emotion[0])
                    if len(result_keys) < 3:
                        remaining_emotions = [k for k in matched_keys_emotion[1:-1] if k != predicted_key_emotion]
                        result_keys.append(random.choice(remaining_emotions))
                else:
                    result_keys.append(matched_keys_emotion[-1])
                    result_keys.append(matched_keys_emotion[0])
                    remaining_emotions = matched_keys_emotion[1:-1]
                    result_keys.append(random.choice(remaining_emotions))
        return result_keys
    
    def get_icon_ids(self, diary_content):
        result_keys = self.get_result_keys(diary_content)
        icon_ids = []

        for key in result_keys:
            if key in self.loader.word_emotion_synonym:
                available_choices = self.loader.match_emotion.get(key, [])
                if available_choices:
                    while True:
                        temp_key = random.choice(available_choices)
                        if temp_key not in icon_ids:
                            icon_ids.append(temp_key)
                            break
            else:
                icon_ids.append(int(key))
        return list(icon_ids)
    
    def get_fonts(self, icon_ids):
        fonts = []
        for id in icon_ids:
            fonts.append(self.loader.icons[self.loader.icons['iconId'] == id]['font'].values[0])
        return fonts
    
    def get_icons(self, diary_content):
        try:
            return self.get_fonts(self.get_icon_ids(diary_content))
        except:
            return False