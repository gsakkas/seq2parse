>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
x = []
for i in range(1:10):
    x.append(i)
-----------------Repaired Program-----------------

>>> Repair #1
x = []
for i in range(1 , 10):
    x.append(i)

>>> pylint: OK!!!

>>> Repair #2
x = []
for i in range(1):
    x.append(i)

>>> pylint: OK!!!

>>> Repair #3
x = []
for i in range(10):
    x.append(i)

>>> pylint: OK!!!
--------------Original Fix Program----------------
x = []
for i in range(1, 10):
    x.append(i)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
