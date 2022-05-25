>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
string = input().split()
print('{:.2f}'.format(string.count('A') / len(string)
-----------------Repaired Program-----------------

>>> Repair #1
string = input().split()
print('{:.2f}'.format(string.count('A') / len(string )) )
>>> pylint: OK!!!

>>> Repair #2
string = input().split()
print('{:.2f}'.format(string.count('A' )) / len(string) )
>>> pylint: OK!!!

>>> Repair #3
string = input().split()
print('{:.2f}'.format(string ).count('A') / len(string) )
>>> pylint: OK!!!
--------------Original Fix Program----------------
string = input().split()
print('{:.2f}'.format(string.count('A') / len(string)))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
