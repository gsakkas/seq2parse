>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
def int_sqrt(25):
    for i in i(n+1):
        if (i * i) <= n:
            number = 1
-----------------Repaired Program-----------------

>>> Repair #1
def int_sqrt(n):
    for i in i(n + 1):
        if(i * i)<= n :
            number = 1

>>> pylint: OK!!!
--------------Original Fix Program----------------
def int_sqrt(n):
    for i in i(n+1):
        if (i * i) <= n:
            number = 1
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
