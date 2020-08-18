'''
dict를 여러가지 방법으로 정렬
'''

mydict = {'a':20, 'b':30, 'c':10}

# 값(value)이 가장 큰 거부터 역순으로 정렬
byValues = sorted(mydict.values(), reverse=True)
print( byValues )

# 키를 기준으로 역순 정렬
byKeys = sorted(mydict.keys(), reverse=True)
print( byKeys )

# 값(value)을 역순으로 정렬하되 키를 보여 주기
keysortByValue = sorted(mydict, key=mydict.get, reverse=True)
print( keysortByValue )