'''
Exercise 2.1
'''
def product_of_unions(A, B, S, T):
    x = A | B
    a_s = {}
    for i in A:
        for n in S:
            a_s.add((i,n))
    at = {}
    for i in A:
        for n in T:
            at.add((i,n))
    bs = {}
    for i in B:
        for n in S:
            bs.add((i,n))
    bt = {}
    for i in B:
        for n in T:
            bt.add((i,n))
    return x, a_s & at & bs & bt

A = {1, 2}
B = {1, 3}
S = {-1, 0}
T = {0, 10}
print(product_of_unions(A, B, S, T))
'''
({1, 2, 3}, {(1, -1),(1, 0),(1, 10),(2, -1),(2, 0),(2, 10),(3, -1),(3, 0),(3, 10)})
'''
assert(product_of_unions(A, B, S, T) == ({1, 2, 3}, {(1, -1),  (1, 0),  (1, 10),  (2, -1),  (2, 0),  (2, 10),  (3, -1),  (3, 0),  (3, 10)}))

A = {5}
B = {5, 6}
S = {-1, 0, 1}
T = {1, 2}
print((product_of_unions(A, B, S, T))

#({5, 6}, {(5, -1), (5, 0), (5, 1), (5, 2), (6, -1), (6, 0), (6, 1), (6, 2)})

       ({5, 6}, {(5, -1), (5, 0), (5, 1), (5, 2), (6, -1), (6, 0), (6, 1), (6, 2)})   )
