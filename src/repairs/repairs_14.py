>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
def square(a)
    b = 1
-----------------Repaired Program-----------------

>>> Repair #1
def square(a) :
    b = 1

>>> pylint: OK!!!

>>> Repair #2
def square(a) :
    b = 1
    ( )
>>> pylint: OK!!!

>>> Repair #3
def square(a) :
    b = 1
    a
>>> pylint: OK!!!
--------------Original Fix Program----------------
def square(a):
    b = 1
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
