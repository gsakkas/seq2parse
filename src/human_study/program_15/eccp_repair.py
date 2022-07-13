n = int(input())
1 <= n <= 20
for test in range(n):
    x = int(input())
    if x % 2 == 0:
        print(str(x) + 'is even')
    else:
        print(str(x) + 'is odd')
