'''
colab에서 실행시 아래 명령어 실행
!apt-get update
!apt-get install g++ openjdk-8-jdk
!pip3 install konlpy
'''
from konlpy.tag import Mecab
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager, rc
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import word2vec
import re
import codecs
from bs4 import BeautifulSoup
from konlpy.tag import Twitter

def get_korean_morphs(words) :
    mecab = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
    # print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))
    return mecab.morphs(words)

def get_korean_clean_word(words, stopwords):
    nouns = []
    tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
    for post in words:
        for noun in tagger.nouns(post):
            if noun not in stopwords:
                nouns.append(noun)  # stopwords에 없을 때, 사전에다 넣어줘라!
    return nouns

def get_korean_pos_tag(words):
    words = []
    tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
    for post in words:
        words.extend(tagger.pos(post))
    return words

def get_korean_noun_list(malist):
    word_dic = {}
    #malist = [('사랑', 'Noun'), ('이', '조사'), ('사랑', 'Noun')] # 얘가 dictionary 타입이여서 0번지, 1번지   형태로 되어 있다.
    for word in malist:
        if word[1] == "Noun":  # 명사 확인하기
            if not (word[0] in word_dic):
                word_dic[word[0]] = 0
            word_dic[word[0]] += 1  # 카운트하기
    keys = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)
    return keys

def sort_by_keys(dict):
    return sorted(value = dict.keys(), reverse=True)

def sort_by_values(dict):
    return sorted(value = dict, key=dict.get, reverse=True)

def get_most_common_words(word_list, num):
    # num_top_nouns = 20
    counter = Counter(word_list)
    top_words = dict(counter.most_common(num))
    return top_words

def draw_word_cloud(word_list):
    wc = WordCloud(background_color="white", font_path='./font/NanumBarunGothic.ttf')
    wc.generate_from_frequencies(word_list)
    figure = plt.figure()
    figure.set_size_inches(10, 10)
    ax = figure.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(wc)

def draw_word_cloud_with_mask(word_list, imagepath):
    mask = np.array(Image.open(imagepath))
    # 워드 클라우드 설정
    wc = WordCloud(background_color="white", mask=mask, contour_width=3,
                         font_path='../font/NanumBarunGothic.ttf')
    wc.generate_from_frequencies(word_list)
    # 이미지 표시
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    # 이미지 저장
    wc.to_file("wordcloud.png")

def draw_bar_graph(word_list, barcount):
    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)

    plt.xlabel('주요 단어')
    plt.ylabel('빈도 수')
    plt.grid(True)
    #barcount = 10  # 10개만 그리겠다.

    Sorted_Dict_Values = sorted(word_list.values(), reverse=True)
    #print(Sorted_Dict_Values)
    plt.bar(range(barcount), Sorted_Dict_Values[0:barcount], align='center')

    Sorted_Dict_Keys = sorted(word_list, key=word_list.get, reverse=True)
    #print(Sorted_Dict_Keys)
    plt.xticks(range(barcount), list(Sorted_Dict_Keys)[0:barcount], rotation='70')

    plt.show()

# bow
def get_count_vector(sentence_list):
    # 1글자도 인식이 되도록 토큰 패턴 변경
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    cv.fit(sentence_list)
    #print(cv.vocabulary_)
    cv_array = cv.transform(sentence_list).toarray()
    return cv_array

# tfidf
def get_tfidf_vector(sentence_list):
    #galexy_tfidv = TfidfVectorizer(stop_words=["스마트폰"]).fit(galexy_top_nouns)
    tfidv = TfidfVectorizer()
    tfidv.fit(sentence_list)
    # print(tfidv.vocabulary_)
    tfidv_array = tfidv.transform(sentence_list).toarray()
    return tfidv_array

def get_smiliarity_by_count_vector(text1, text2):
    cv1 = get_count_vector(text1)
    cv2 = get_count_vector(text2)
    cosine_sim = cosine_similarity(cv1, cv2)
    return cosine_sim

def get_smiliarity_by_tfidf_vector(text1, text2):
    tv1 = get_tfidf_vector(text1)
    tv2 = get_tfidf_vector(text2)
    cosine_sim = linear_kernel(tv1, tv2)
    return cosine_sim

def get_most_similar_items(item, series, num):
   return

def create_word2vec_model(words):
    data = word2vec.LineSentence(words)
    model = word2vec.Word2Vec(data, size=200, window=10, hs=1, min_count=2, sg=1)
    model_name = 'word2vec.model'
    model.save(model_name)
    print('파일 ', model_name, '저장 완료')

def draw_word2vec_bar_graph(words, x_start, x_end):
    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)
    su = len(words)  # 전체 데이터 수
    # 축에 보여질 항목 이름들
    item = list(item[0] for item in words)
    # 그려지는 수치 데이터
    count = list(item[1] for item in words)
    plt.barh(range(su), count, align='center')
    plt.yticks(range(su), item, rotation='10')
    plt.xlim(x_start, x_end)
    plt.grid(True)
    plt.show()

def get_most_similar_item_by_word2vec(model, item_list) :
    #model.wv.most_similar(positive=["한국", "트럼프"], negative=["미국"]))
    return model.wv.most_similar(positive=item_list)

def get_email_address(text) :
    return re.search(r'[\w.-]+@[\w.-]+', text).group()

def print_word_similarity_by_word2vec(word_list, embed) :
    #word_list = ["coffee", "cafe", "football", "soccer"]
    embeddings = embed(word_list)
    for i in range(len(word_list)):
        for j in range(i, len(word_list)):
            print("(", word_list[i], ",", word_list[j], ")", np.inner(embeddings[i], embeddings[j]))

def load_word2vec_wiki250_model() :
    # pip install tensorflow_hub
    # gpu 사용하므로 colab에서 사용 권장
    # import tensorflow_hub as hub
    # hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")
    return

def get_review_text(text):
    return re.sub("[^a-zA-Z]", text).group()


def get_most_similar_items(index, series, cos_sim, num):

    scores = list(enumerate(cos_sim[index])) # 이뉴머레이트는 각각의 값에다가 index를 달아주는 것을 말한다.   => 이걸 list로 만들어준다는 코드이다.
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    scores = scores[1:num]
    indices = [x[0] for x in scores]

    ############이 코드 못 끝냄 ㅠㅠ 보러가다!!!

def draw_histogram_by_token(token):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.hist(token, bins=50, alpha=0.5, color="r", label="word")
   # plt.hist(review_len_by_eumjeol, bins=50, alpha=0.5, color="b", label="alphabet")

    plt.yscale('log', nonposy='clip')
    plt.title('Review Length Histogram')
    plt.xlabel('review length')
    plt.ylabel('number of reviews')


def print_token_info(token):
    import numpy as np
    print('문장 최대 길이 : {}'.format(np.max(token)))
    print('문장 최소 길이 : {}'.format(np.min(token)))
    print('문장 평균 길이 : {:.2f}'.format(np.mean(token)))
    print('문장 길이 표준편타 : {:.2f}'.format(np.std(token)))
    print('문장 중간 길이 : {}'.format(np.median(token)))
    print('제 1사분위 길이 : {}'.format(np.percentile(token, 25)))  # percentile은 4분위 수를 의미한다.
    print('제 3사분위 길이 : {}'.format(np.percentile(token, 75)))

def draw_count_plot(data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sentiment = data.value_counts()
    #sentiment = train_df['sentiment'].value_counts()
    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.countplot(data)
