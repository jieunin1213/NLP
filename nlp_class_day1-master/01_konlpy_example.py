'''
1. colab에서 실행시 아래 명령어 실행
!apt-get update
!apt-get install g++ openjdk-8-jdk
!pip3 install konlpy

2. colab에서 MeCab 설시 방법 참고
https://colab.research.google.com/drive/1tL2WjfE0v_es4YJCLGoEJM5NXs_O_ytW#scrollTo=1lxZgy_vjaah

3. https://konlpy.org/ko/latest/api/konlpy.tag/
API 예제 실행

4. https://konlpy.org/ko/latest/data/#corpora
말뭉치

5. https://konlpy.org/ko/latest/examples/
사용 예시
'''

# Hannanum Class
from konlpy.tag import Hannanum
hannanum = Hannanum()
print(hannanum.analyze(u'롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.'))

#Kkma Class
from konlpy.tag import Kkma
kkma = Kkma()
print(kkma.morphs(u'공부를 하면할수록 모르는게 많다는 것을 알게 됩니다.'))

# Komoran Class
from konlpy.tag import Komoran
komoran = Komoran()
print(komoran.morphs(u'우왕 코모란도 오픈소스가 되었어요'))

# MeCab installation needed
from konlpy.tag import Mecab
mecab = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))

# Twitter Class
# from konlpy.tag import Twitter
# twitter = Twitter()
# print(twitter.morphs(u'단독입찰보다 복수입찰의 경우'))

from konlpy.tag import Okt
twitter = Okt()
print(twitter.morphs(u'단독입찰보다 복수입찰의 경우'))
