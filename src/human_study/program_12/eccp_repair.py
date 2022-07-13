a = int(input())
b = int(input())
s = 1
while True:
    if s % a != 0 and s % b != 0:
        s += 1
    else:
        break
print(s)
