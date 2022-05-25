>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
a = [99, 88, 33]
def modify(b):
b += [12, 55]
print(b)
print(a)
modify(a)
print(a)
-----------------Repaired Program-----------------

>>> Repair #1
a = [99, 88, 33]
def modify(b) :
    b += [12, 55]
    print(b)
    print(a)
    modify(a)
    print(a)

>>> pylint: OK!!!

>>> Repair #2
a = [99, 88, 33]
def modify(b) : b
b += [12, 55]
print(b)
print(a)
modify(a)
print(a)
>>> pylint: OK!!!

>>> Repair #3
a = [99, 88, 33]
def modify(b) :
    b += [12, 55]
print(b)
print(a)
modify(a)
print(a)
>>> pylint: OK!!!
--------------Original Fix Program----------------
a = [99, 88, 33]
def modify(b):
    b += [12, 55]
print(b)
print(a)
modify(a)
print(a)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
