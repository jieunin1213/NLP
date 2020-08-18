'''
말뭉치 파일 단어 빈도 분석
- 자료형 구조에 대한 이해
- list, tuple, dict 사용하기
'''
word_dic = {}
malist = [('사랑', 'Noun') , ('이', '조사'), ('사랑', 'Noun')]
for word in malist:
    if word[1] == "Noun": #  명사 확인하기 --- (※3)
        if not (word[0] in word_dic):
            word_dic[word[0]] = 0
        word_dic[word[0]] += 1 # 카운트하기


print(word_dic)

# 값의 수치가 가장 큰 것 부터 역순으로 정렬하여 보여 준다.
keys = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)
print(keys)