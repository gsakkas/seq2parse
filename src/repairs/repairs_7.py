>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
def square(3):
    return y**y

print(square(2))
-----------------Repaired Program-----------------

>>> Repair #1
def square(y):
    return y ** y
print(square(2))
>>> pylint: OK!!!
--------------Original Fix Program----------------
def square(y):
    return y**y

print(square(2))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
