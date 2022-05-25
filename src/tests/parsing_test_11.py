

def sum_evens_2d(xss):
    sum_even = 0
    for i in range(len(xss)):
        for j in range(len(xss[i])):
            if xss[0][:]%2 == 0:
                if xss[1][:]%2 == 0:
                    sum_even += sum_even
                return sum_even

sum_evens_2d([[1,2,3],[4,5,6]])
