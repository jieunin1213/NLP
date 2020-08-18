'''
어휘가 다른 문서에는 별로 등장하지 않고 특정 문서에만 집중적으로 등장할 때 그 어휘야말로 실질적으로 그 문서의 주제를 잘 담고 있는 핵심어라 할 수 있다.
'''
import pandas as pd
df = pd.read_csv('./data/smartphone.csv', encoding='utf-8')
galexy_posts = df.get('Title') + " " + df.get('Description')

from konlpy.tag import Mecab
tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")

galexy_stop_words = "은 이 것 등 더 를 좀 즉 인 옹 때 만 원 이때 개 일 기 시 럭 갤 성 삼 스 폰 트 드 기 이 리 폴 사 전 마 자 플 블 가 중 북 수 팩 년 월 저 탭"
galexy_stop_words = galexy_stop_words.split(' ')
galexy_stop_words[0:10]

# 불용어 제거
galexy_nouns = []
for post in galexy_posts:
    for noun in tagger.nouns(post):
        if noun not in galexy_stop_words:
            galexy_nouns.append(noun)

galexy_nouns[0:10]

from collections import Counter
num_top_nouns = 20
galexy_nouns_counter = Counter(galexy_nouns)
galexy_top_nouns = dict(galexy_nouns_counter.most_common(num_top_nouns))
print(galexy_top_nouns)

from sklearn.feature_extraction.text import TfidfVectorizer
galexy_tfidv = TfidfVectorizer().fit(galexy_top_nouns)
galexy_tfidv.transform(galexy_posts).toarray()

print(pd.DataFrame(galexy_tfidv.transform(galexy_posts).toarray()))