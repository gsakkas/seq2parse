>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
from random import *
r = []
for i in [10, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    r.append(randint(1, i))
    sum = 0
    for num in r:
    sum = sum + num
print(sum)
-----------------Repaired Program-----------------

>>> Repair #1
from random import *
r = []
for i in [10, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    r.append(randint(1, i))
    sum = 0
    for num in r :
        sum = sum + num
    print(sum)

>>> pylint: OK!!!

>>> Repair #2
from random import *
r = []
for i in [10, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    r.append(randint(1, i))
    sum = 0
    for num in r :
        sum = sum + num
print(sum)
>>> pylint: OK!!!

>>> Repair #3
from random import *
r = []
for i in [10, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    r.append(randint(1, i))
    sum = 0
    for num in r : ( )
    sum = sum + num
print(sum)
>>> pylint: OK!!!
--------------Original Fix Program----------------
from random import *
r = []
for i in [10, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    r.append(randint(1, i))
    sum = 0
for num in r:
    sum = sum + num
print(sum)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
