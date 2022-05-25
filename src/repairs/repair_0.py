a = [99]
def modify(b):
    b += [12]
    modify(a)
    print(a)
