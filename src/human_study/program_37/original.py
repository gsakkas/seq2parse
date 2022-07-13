def m(a, b):
    j = a
    i = b
    l = []
    while len(j) != 0 and len(i) != 0:
        if len(j) == 0:
            l.append(i)
        elif len(i) == 0:
            l.append(j)
        elif j[0] > i[0]:
            l.append(i[0])
            i.remove(i[0]
        elif i[0] > j[0]:
            l.append(j[0])
            i.remove(j[0])
    return l

t = [1, 2, 3, 4, 5, 6, 7]
k = [2, 2, 4, 5, 8, 9, 9]
print(m(t, k))
