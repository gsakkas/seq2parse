def divvy(seq, number):
    less = 0
    more = 0
    for item in seq:
        if item < number:
            less += 1
    elif item > number:
        more += 1
    else:
        return None
    return less + more

print(divvy([5, -2, 0, -2, 1], -2))
