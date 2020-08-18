'''
국립 국어원에서 제공하는 말뭉치 파일인 '박경리의 토지'에 대하여
단어들의 출현 빈도에 대하여 분석
텍스트 문서의 내용을 읽어서 형태소(품사단위) 분석을 수행
사용된 빈도수가 많은 항목에 대해 워드 클라우스 생성
https://ithub.korean.go.kr/user/total/database/corpusManager.do
'''
import codecs
import pytagcloud
import webbrowser
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
from bs4 import BeautifulSoup
from konlpy.tag import Twitter

# utf-16 인코딩으로 파일을 열고 글자를 출력하기 --- (※1)
fp = codecs.open("./data/BEXX0003.txt", "r", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")
body = soup.select_one("body > text")
text = body.getText()
print(text)

# 텍스트를 한 줄씩 처리하기 --- (※2)
twitter = Twitter()

word_dic = {}

lines = text.split("\r\n")

for line in lines:
    malist = twitter.pos(line)
    for word in malist:
        if word[1] == "Noun":  # 명사 확인하기 --- (※3)
            if not (word[0] in word_dic):
                word_dic[word[0]] = 0
            word_dic[word[0]] += 1  # 카운트하기

# 많이 사용된 명사 출력하기 --- (※4)
keys = sorted(word_dic.items(), key=lambda x: x[1], reverse=True)


def saveWordCloud(wordInfo):
    taglist = pytagcloud.make_tags(dict(wordInfo).items(), maxsize=80)
    print(type(taglist))  # <class 'list'>
    filename = 'wordcloud.png'

    pytagcloud.create_tag_image(taglist, filename, \
                                size=(640, 480), fontname='korean', rectangular=False)
    webbrowser.open(filename)


def showGraph(wordInfo):
    font_location = 'c:/Windows/fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=font_location).get_name()
    matplotlib.rc('font', family=font_name)

    plt.xlabel('주요 단어')
    plt.ylabel('빈도 수')
    plt.grid(True)

    barcount = 10  # 10개만 그리겠다.

    Sorted_Dict_Values = sorted(wordInfo.values(), reverse=True)
    print(Sorted_Dict_Values)
    print('dd')
    plt.bar(range(barcount), Sorted_Dict_Values[0:barcount], align='center')

    Sorted_Dict_Keys = sorted(wordInfo, key=wordInfo.get, reverse=True)
    print(Sorted_Dict_Keys)
    plt.xticks(range(barcount), list(Sorted_Dict_Keys)[0:barcount], rotation='70')

    plt.show()


wordInfo = dict()
for word, count in keys[:500]:
    print("{0}({1}) ".format(word, count), end="")
    if (count > 60) and len(word) >= 2:
        wordInfo[word] = count

saveWordCloud(wordInfo)
showGraph(wordInfo)

print('작업 종료')