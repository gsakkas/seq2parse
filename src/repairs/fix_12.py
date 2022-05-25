def primes(n):
    primes = []
    i = 0
    while i < n:
        if i/n == int(i/n) and i == n:
            primes.append(i)
    return primes

primes(23)
