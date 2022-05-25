>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
old = [1,2,3,4,5]
>>> new = old
>>> old = [6]
>>> print(new)
-----------------Repaired Program-----------------

>>> Repair #1
old = [1, 2, 3, 4, 5]
new = old
old = [6]
print(new)
>>> pylint: OK!!!
--------------Original Fix Program----------------
old = [1,2,3,4,5]
new = old
old = [6]
print(new)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

COST 3!!!
