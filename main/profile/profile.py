import re
from konlpy.tag import Okt
from wordcloud import WordCloud
from collections import Counter

def create_wordcloud(diarys, file_name):
    # 분석할 데이터 추출
    try:
        text = re.sub(r'[^\w\n]', ' ', diarys)
        stopwords = [
                '오늘', '아주', '조금', '매우', '다음', '그때', '너무', '같이', '정말', '모두', '하루', '처음', '시간', '기분', '우리'
        ]

        # 품사 태깅 : 명사추출
        okt = Okt()
        nouns = okt.nouns(text)
        nouns = [noun for noun in nouns if noun not in stopwords]
    
        count = Counter(nouns)
    
        wordCount = dict()
        for tag, counts in count.most_common(30):
            if(len(str(tag)) > 1):
                wordCount[str(tag)] = counts
    
        fontPath = 'C:/windows/fonts/hmkmmag.ttf'
        
        wc = WordCloud(fontPath, background_color='white', width=800, height=600, max_words=25)
        wordcloud = wc.generate_from_frequencies(wordCount)
    
        wordcloud.to_file('./images/profile/' + file_name)
    
        return True
    except:
        return False