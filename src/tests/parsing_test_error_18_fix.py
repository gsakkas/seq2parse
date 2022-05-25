b = []
a = [int(i) for i in input().split()]
k = len(a)
for i in a:
    if i == 0:
        j = a[i+1]
        m = i(k)
    j = a[i+1]
    m = a[i-1]
    b.append(j+m)
print(b)

