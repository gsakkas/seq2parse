n = int(input())
s = n
q = 0 + abs(n)**2
while s != 0:
    n = int(input())
    s = s + n
    q = q + abs(n)**2
    if s == 0:
        break
print(q)
