'''
의미 연결망 분석
주요 단어 20개를 가지고 의미 연결망을 만들어 중심성 지수를 구하기
'''

# 정규화 라이브러리
import re
# 의미 연결망 분석
import networkx as nx
import pandas as pd
df = pd.read_csv('./data/smartphone.csv', encoding='utf-8')
galexy_posts = df.get('Title') + " " + df.get('Description')
galexy_post_date = df.get('Post Date')

from konlpy.tag import Mecab
tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")

galexy_stop_words = "은 이 것 등 더 를 좀 즉 인 옹 때 만 원 이때 개 일 기 시 럭 갤 성 삼 스 폰 트 드 기 이 리 폴 사 전 마 자 플 블 가 중 북 수 팩 년 월 저 탭"
galexy_stop_words = galexy_stop_words.split(' ')
galexy_stop_words[0:10]

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

galexy_sentences = []
# 블로그 내용에 대해서 문장으로 나누기 위해서 문장의 끝을 나타내는 . ; ? ! 를 구분자로 사용
for post in galexy_posts:
    galexy_sentences.extend(re.split('; |\.|\?|\!', post))
galexy_sentences[0:10]


#블로그 내용을 문장별로 구분하고 구분된 문당별로 명사를 추출하여 정리
galexy_sentences_nouns = []
for sentence in galexy_sentences:
    sentence_nouns = tagger.nouns(sentence)
    galexy_sentences_nouns.append(sentence_nouns)
galexy_sentences_nouns[0:10]

# 상위 단어 top_nouns에 대해서 key에 해당하는 단어, value에 해당하는 id를 넣어 딕셔너리 형태(word2id)로 저장
galexy_word2id = {w: i for i, w in enumerate(galexy_top_nouns.keys())}
galexy_word2id

# 상위 단어 top_nouns에 대해서 key에 해당하는 단어, id, value에 해당하는 단어를 넣어 딕셔너리 형태(id2word)로 저장
galexy_id2word = {i: w for i, w in enumerate(galexy_top_nouns.keys())}
galexy_id2word

# 상위 단어들에 대해서 그 개수만큼의 인접 행렬을 만들고, 문장 내에 상위 단어가 함께 포함된 비중에 따라 가중치를 계산하여 행렬에 표현
# 인접행렬을 생성하기 위해서 이전에 생성해둔 word2id를 이용
# 행렬에서 만약 가중치가 0이상이면 서로 연결되어 있음을 의미
import numpy as np
galexy_adjacent_matrix = np.zeros((num_top_nouns, num_top_nouns), int)
for sentence in galexy_sentences_nouns:
    for wi, i in galexy_word2id.items():
        if wi in sentence:
            for wj, j in galexy_word2id.items():
                if i != j and wj in sentence:
                    galexy_adjacent_matrix[i][j] += 1
galexy_adjacent_matrix


# 인접 행렬로 연결망을 만들고 연결망에 포함된 인접 행렬에 대한 결과를 살펴본다
galexy_network = nx.from_numpy_matrix(galexy_adjacent_matrix)
list(galexy_network.adjacency())

# 생성된 연결망 데이터를 시각화하여 나타낸다
# 여기서 labels 값으로 이전에 생성한 id2word를 이용하고 그래프에 표시될 한글폰트를 설치
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import rc

font_path="./font/NanumBarunGothic.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

fig = plt.figure()
fig.set_size_inches(20, 20)
ax = fig.add_subplot(1, 1, 1)
ax.axis("off")
option = {
    'node_color' : 'lightblue',
    'node_size' : 2000,
    'size' : 2
}
nx.draw(galexy_network, labels=galexy_id2word, font_family=font_name, ax=ax, **option)


# 의미 연결망을 시각화하는 표현 방법
# Random Layout, Circular Layout, Spectral Layout, Spring Layout형태로 시각화 하여 표현

fig = plt.figure()
fig.set_size_inches(20, 20)
option = {
    'node_color' : 'lightblue',
    'node_size' : 500,
    'size' : 100
}

plt.subplot(221)
plt.title('Random Layout', fontsize=20)
nx.draw_random(galexy_network, labels=galexy_id2word, font_family=font_name, **option)
plt.subplot(222)
plt.title('Circular Layout', fontsize=20)
nx.draw_circular(galexy_network, labels=galexy_id2word, font_family=font_name, **option)
plt.subplot(223)
plt.title('Spectral Layout',fontsize=20)
nx.draw_spectral(galexy_network, labels=galexy_id2word, font_family=font_name, **option)
plt.subplot(224)
plt.title('Spring Layout',fontsize=20)
nx.draw_spring(galexy_network, labels=galexy_id2word, font_family=font_name, **option)

plt.show()

#Degree (연결중심성)
nx.degree_centrality(galexy_network)

#Eigenvector (위세 중심성)
nx.eigenvector_centrality(galexy_network, weight='weight')

#Closeness (근접 중심성)
nx.closeness_centrality(galexy_network, distance='weight')

#Current Flow Closeness (매개 중심성)
nx.current_flow_closeness_centrality(galexy_network)

#Current Flow Betweenness
nx.current_flow_betweenness_centrality(galexy_network)

#Communicability Betweenness
nx.communicability_betweenness_centrality(galexy_network)