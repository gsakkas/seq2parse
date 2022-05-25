>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
-------------Original Buggy Program---------------

def primes(n: int) -> list:
    """gives all primes <=n
    """
    primes = [2]
    pot_primes = list(range(2,n+1))
    for number in pot_primes:
        for prime in primes:
            if number // prime == 0:
            continue
        else:
            primes +=  [x]


    print (primes)

primes (20)
-----------------Repaired Program-----------------

>>> Repair #1

def primes(n : int) -> list :
    """gives all primes <=n
    """
    primes = [2]
    pot_primes = list(range(2, n + 1))
    for number in pot_primes :
        for prime in primes :
            if number // prime == 0 :
                continue
            else :
                primes += [x]
        print(primes)
    primes(20)


>>> Repair #2

def primes(n : int) -> list :
    """gives all primes <=n
    """
    primes = [2]
    pot_primes = list(range(2, n + 1))
    for number in pot_primes :
        for prime in primes :
            if number // prime == 0 :
                continue
            else :
                primes += [x]
        print(primes)
primes(20)

>>> Repair #3

def primes(n : int) -> list :
    """gives all primes <=n
    """
    primes = [2]
    pot_primes = list(range(2, n + 1))
    for number in pot_primes :
        for prime in primes :
            if number // prime == 0 :
                continue
            else :
                primes += [x]
    print(primes)
primes(20)
--------------Original Fix Program----------------

def primes(n: int) -> list:
    """gives all primes <=n
    """
    primes = [2]
    pot_primes = list(range(2,n+1))
    for number in pot_primes:
        for prime in primes:
            if number // prime == 0:
                continue
        else:
            primes +=  [x]


    print (primes)

primes (20)
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
