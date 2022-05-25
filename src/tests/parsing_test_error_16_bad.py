global res = ""
def inverse(n):
    n = str(n)
    if len(n) == 1:
        return n
    else:
        res += n[len(n)-1:]
        n = n[:len(n)-1]
        inverse(n)
    return res


print(inverse(574))
