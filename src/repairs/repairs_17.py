>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
from random import
r = []
for i in [10, 200,300,400,500,600,700,800,900,1000]:
    r.append(randint(1, i))
-----------------Repaired Program-----------------

>>> Repair #1
from random import *
r = []
for i in [10, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    r.append(randint(1, i))

>>> pylint: OK!!!
--------------Original Fix Program----------------
from random import *
r = []
for i in [10, 200,300,400,500,600,700,800,900,1000]:
    r.append(randint(1, i))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

FAILED! Didn't have the erule
