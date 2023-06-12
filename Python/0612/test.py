# zip
a = [1,2,3]
b = ['a', 'b', 'c']
c = ['#', '$', '!']
result = zip(a,b,c)
for i in result :
    print(i)

list_temp = list(zip(a,b,c))
print(list_temp)