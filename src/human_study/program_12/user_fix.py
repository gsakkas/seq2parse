a = int(input())
b = int(input())
s = 1
while s % a != 0 and s % b != 0:
    s += 1
    break
print(s)
