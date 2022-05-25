>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
a = 5
if 0 =< a < 10:
    print("test passé")
-----------------Repaired Program-----------------

>>> Repair #1
a = 5
if 0 < a < 10 :
    print("test passé")

>>> pylint: OK!!!

>>> Repair #2
a = 5
if 0 < a < 10 :
    print("test passé")
    a
>>> pylint: OK!!!

>>> Repair #3
a = 5
if 0 < a < 10 :
    print("test passé")
    print
>>> pylint: OK!!!
--------------Original Fix Program----------------
a = 5
if 0 < a < 10:
    print("test passé")
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
