>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
a=(input("enter a number"))
i=0
while i<(len(a-2)):
    if a[0]==0 and a[1]==1 and a[i]+a[i+1]=a[i+2]:
        i=i+i
        print("y")
    else:
        print("n")
-----------------Repaired Program-----------------

>>> Repair #1
a =(input("enter a number"))
i = 0
while i <(len(a - 2)) :
    if a [0]== 0 and a [1]== 1 and a [i]+ a [i + 1]and a [i + 2]:
        i = i + i
        print("y")
    else :
        print("n")

>>> pylint: OK!!!

>>> Repair #2
a =(input("enter a number"))
i = 0
while i <(len(a - 2)) :
    if a [0]== 0 and a [1]== 1 and a [i]+ a [i + 1]< a [i + 2]:
        i = i + i
        print("y")
    else :
        print("n")

>>> pylint: OK!!!

>>> Repair #3
a =(input("enter a number"))
i = 0
while i <(len(a - 2)) :
    if a [0]== 0 and a [1]== 1 and a [i]+ a [i + 1]< a [i + 2]:
        i = i + i
        print("y")
    else :
        print("n")

>>> pylint: OK!!!
--------------Original Fix Program----------------
a=(input("enter a number"))
i=0
while i<(len(a-2)):
    if a[0]==0 and a[1]==1 and a[i]+a[i+1]==a[i+2]:
        i=i+i
        print("y")
    else:
        print("n")
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
