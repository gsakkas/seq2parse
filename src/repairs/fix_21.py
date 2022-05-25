def foo(x):
    z = 0

    while x > 0:
        x = x - 2
        a = x * 2

        while a > x:
            a = a - 1
            z = z + 1

    return z

x = foo(5)
print(x)
