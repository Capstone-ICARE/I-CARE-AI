import re
from konlpy.tag import Okt
from wordcloud import WordCloud
from collections import Counter

class Profile:
    def __init__(self):
        self.okt = Okt()
        
    def create_profile(self, diary, file_name):
        try:
            text = re.sub(r'[^\w\n]', ' ', diary)
            stopwords = [
                    '오늘', '아주', '조금', '매우', '다음', '그때', '너무', '같이', '정말', '모두', '하루', '처음', '시간', '기분', '우리'
                    ]

            # 품사 태깅 : 명사추출
            nouns = self.okt.nouns(text)
            nouns = [noun for noun in nouns if noun not in stopwords]
            
            count = Counter(nouns)
            
            wordCount = dict()
            for tag, counts in count.most_common(30):
                if(len(str(tag)) > 1):
                    wordCount[str(tag)] = counts
    
            fontPath = 'C:/windows/fonts/hmkmmag.ttf'
        
            wc = WordCloud(fontPath, background_color='white', width=400, height=300, max_words=25)
            profile = wc.generate_from_frequencies(wordCount)
    
            profile.to_file('./images/profile/' + file_name)
    
            return True
        except:
            return False