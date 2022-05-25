def primes(n: int) -> list:
    primes = [2]
    pot_primes = list(range(2, n+1))
    for number in pot_primes:
        for prime in primes:
            if number // prime == 0:
            continue
        else:
            primes += [x]
    print(primes)

primes(20)
