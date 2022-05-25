>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
def foo(x):
    z = 0

    while x  0:
        x = x - 2
        a = x * 2

        while a  x:
            a = a - 1
            z = z + 1

    return z

x = foo(5)
print(x)
-----------------Repaired Program-----------------

>>> Repair #1
def foo(x) :
    z = 0
    while x < 0 :
        x = x - 2
        a = x * 2
        while a ( x ) :
            a = a - 1
            z = z + 1
    return z
x = foo(5)
print(x)
>>> pylint: OK!!!

>>> Repair #2
def foo(x) :
    z = 0
    while x :
        x = x - 2
        a = x * 2
        while a ( x ) :
            a = a - 1
            z = z + 1
    return z
x = foo(5)
print(x)
>>> pylint: OK!!!

>>> Repair #3
def foo(x) :
    z = 0
    while 0 :
        x = x - 2
        a = x * 2
        while a ( x ) :
            a = a - 1
            z = z + 1
    return z
x = foo(5)
print(x)
>>> pylint: OK!!!
--------------Original Fix Program----------------
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
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
