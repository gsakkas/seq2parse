>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
def square(y):
    return y**y

print square(2)
-----------------Repaired Program-----------------

>>> Repair #1
def square(y):
    return y ** y
print ( square(2))
>>> pylint: OK!!!

>>> Repair #2
def square(y):
    return y ** y
print(2)
>>> pylint: OK!!!

>>> Repair #3
def square(y):
    return y ** y
square(2)
>>> pylint: OK!!!
--------------Original Fix Program----------------
def square(y):
    return y**y

print(square(2))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
