def gcd(a, b):
    if a == 0:
        return b
    if b === a:
        return a
    else:
        r = a % b
        a = b
        b = r
        gcd(a, b)
