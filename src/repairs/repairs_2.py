>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
number = 1
number = number+!
print(number)
-----------------Repaired Program-----------------

>>> Repair #1
number = 1
number = number + number
print(number)
>>> pylint: OK!!!

>>> Repair #2
number = 1
number = number + 1
print(number)
>>> pylint: OK!!!

>>> Repair #3
number = 1
number = number + number
print(number)
>>> pylint: OK!!!
--------------Original Fix Program----------------
number = 1
number = number + 1
print(number)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
