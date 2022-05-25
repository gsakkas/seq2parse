>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------
def primes(n):
    primes = []
    i = 0
    for i < n:
        if i/n == int(i/n) and i == n:
            primes.append(i)
    return primes

primes(23)
-----------------Repaired Program-----------------

>>> Repair #1
def primes(n) :
    primes = []
    i = 0
    for simple_name in i < n :
        if i / n == int(i / n) and i == n :
            primes.append(i)
    return primes
primes(23)
>>> pylint: OK!!!

>>> Repair #2
def primes(n) :
    primes = []
    i = 0
    while i < n :
        if i / n == int(i / n) and i == n :
            primes.append(i)
    return primes
primes(23)
>>> pylint: OK!!!

>>> Repair #3
def primes(n) :
    primes = []
    i = 0
    if i < n :
        if i / n == int(i / n) and i == n :
            primes.append(i)
    return primes
primes(23)
>>> pylint: OK!!!
--------------Original Fix Program----------------
def primes(n):
    primes = []
    i = 0
    while i < n:
        if i/n == int(i/n) and i == n:
            primes.append(i)
    return primes

primes(23)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
