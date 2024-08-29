import pandas as pd
import json

icons = pd.read_csv('newIcon.csv')
object_json = 'match_object.json'
emotion_json = 'match_emotion.json'

object_icons = icons[icons['category'] != 'A'][['iconId', 'name']]
objects = dict()
for _, row in object_icons.iterrows():
    iconId = row['iconId']
    name = row['name']
    objects[iconId] = name
    
with open(object_json, 'w', encoding='utf-8') as file:
    json.dump(objects, file, ensure_ascii=False, indent=4)
    
    
emotion_icons = icons[icons['category'] == 'A'][['iconId', 'name']]
emotions = dict()
for _, row in emotion_icons.iterrows():
    iconId = row['iconId']
    name = row['name']
    if name == "재밌다":
        name = 1
    elif name == "사랑하다":
        name = 2
    elif name == "놀라다":
        name = 3
    elif name == "긴장하다":
        name = 4
    elif name == "실망하다":
        name = 5
    elif name == "화나다":
        name = 6
    elif name == "슬프다":
        name = 7
    elif name == "피곤하다":
        name = 8
    elif name == "민망하다":
        name = 9
    elif name == "아쉽다":
        name = 10
    elif name == "아프다":
        name = 11
    if name not in emotions.keys():
        emotions[name] = [iconId]
    else:
        emotions[name].append(iconId)
       
with open(emotion_json, 'w', encoding='utf-8') as file:
    json.dump(emotions, file, ensure_ascii=False, indent=4)