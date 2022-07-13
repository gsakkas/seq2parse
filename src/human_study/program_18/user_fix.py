def fib(x):
    if x == 1 or x == 2:
        return 1
    else:
        a = 1
        b = 1
        for i in range(3, x+1):
            a, b = b, a + b
            return a

fib(5)
