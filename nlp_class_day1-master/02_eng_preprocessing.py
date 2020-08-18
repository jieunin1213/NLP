'''
한국과 글로벌 기업의 영문 지속가능 경영 보고서 380개에 수록된 회장 인사말 모음
'''
import pandas as pd
df = pd.read_csv('./data/CEO3.csv', encoding='latin')
print(df.head())

#회장 인사말 불러오기
texts = df['text']
print(texts.head())

from nltk.tokenize import RegexpTokenizer
# + : 영문자, 숫자, 공백을 제외한 모든 문자를 제거한 뒤에 토큰화
tokenizer = RegexpTokenizer(r'\w+')

from nltk.stem import WordNetLemmatizer
# WordNet 사전에 들어간 단어를 기반으로 표제어추출(lemmatizing) 진행
lemmatizer = WordNetLemmatizer()

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
# nltk 라이브러리에서 제공하는 영어 불용어 다운로드
stop_words = nltk.corpus.stopwords.words('english')
stop_words[0:10]

# nltk 라이브러리에 포함된 wordnet 라이브러리를 이용하여
# 각 영문 텍스트에 대해서 토큰화(tokenizer)와 표제어 추출(lemmatizer) 작업 수행
# 토큰 중에서 불용어에 포함되지 않은 텍스트에 대해서만 추출하여 저장
from nltk import wordnet
nltk.download('wordnet')
preprocessed_texts = []
for text in texts.values:
    tokenized_text = tokenizer.tokenize(text.lower())
    lemmatized_text = [lemmatizer.lemmatize(token) for token in tokenized_text]
    stopped_text = [token for token in lemmatized_text if token not in stop_words]
    preprocessed_texts.append(stopped_text)


# 문서에서 추출한 각 토큰의 개수를 파악
# 전처리된 텍스트로부터 토큰의 개수를 파악하여
# 상위 20개의 토큰에 대해서만 출현 빈도 출력하여 결과 확인
from collections import Counter
tokens = []
for text in preprocessed_texts:
    tokens.extend(text)

counted_tokens = Counter(tokens)
top_20 = counted_tokens.most_common(20)
print(top_20)