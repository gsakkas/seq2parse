>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
a = 5
if a == 0 OR 0 < a < 10:
    print("test passé")
-----------------Repaired Program-----------------

>>> Repair #1
a = 5
if a == 0 < 0 < a < 10 :
    print("test passé")

>>> pylint: OK!!!

>>> Repair #2
a = 5
if a == 0 < a < 10 :
    print("test passé")

>>> pylint: OK!!!

>>> Repair #3
a = 5
if a == 0 < 0 < a < 10 :
    print("test passé")

>>> pylint: OK!!!
--------------Original Fix Program----------------
a = 5
if a == 0 or 0 < a < 10:
    print("test passé")
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
