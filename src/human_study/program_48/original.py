def f(x, y):
    return x - y

def g(x, y):
return f(x + y, 6)

a = 3
b = 4
c = g(a, b)
print(c)
