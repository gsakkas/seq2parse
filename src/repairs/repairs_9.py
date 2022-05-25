>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
d = {'name': 'Liv', 'age': '47'}
k = d.keys()
print(k)
print(type(k)))
v = d.values()
print(v)
print(type(v)))
-----------------Repaired Program-----------------

>>> Repair #1
d = { 'name' : 'Liv', 'age' : '47' }
k = d.keys()
print(k)
print(type(k)())
v = d.values()
print(v)
print(type(v)())
>>> pylint: OK!!!

>>> Repair #2
d = { 'name' : 'Liv', 'age' : '47' }
k = d.keys()
print(k)
print ((type(k)))
v = d.values()
print(v)
print(type(v)())
>>> pylint: OK!!!

>>> Repair #3
d = { 'name' : 'Liv', 'age' : '47' }
k = d.keys()
print(k)
print(( type(k)))
v = d.values()
print(v)
print(type(v)())
>>> pylint: OK!!!
--------------Original Fix Program----------------
d = {'name': 'Liv', 'age': '47'}
k = d.keys()
print(k)
print(type(k))
v = d.values()
print(v)
print(type(v))
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

~10 min!
