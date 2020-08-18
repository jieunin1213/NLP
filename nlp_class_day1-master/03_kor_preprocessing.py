import pandas as pd
df = pd.read_csv('./data/smartphone.csv', encoding='utf-8')
df[0:10]
df.info()

#타이틀과 포스트 설명만 가져오기
posts = df.get('Title') + " " + df.get('Description')
posts[0:10]

from konlpy.tag import Mecab
tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")

stop_words = "은 이 것 등 더 를 좀 즉 인 옹 때 만 원 이때 개 일 기 시 럭 갤 성 삼 스 폰 트 드 기 이 리 폴 사 전 마 자 플 블 가 "
stop_words=stop_words.split(' ')
stop_words[0:10]

nouns = []
for post in posts:
    for noun in tagger.nouns(post):
        if noun not in stop_words:
            nouns.append(noun)
print(nouns[0:10])


words = []
for post in posts:
    words.extend(tagger.pos(post))
words[0:10]