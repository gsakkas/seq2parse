def f(a, b):
    print("a: ", a)
    print("b: ", b)
    
def g(L):
    L[0] = L[0] - 1 
    print (L) 
    return L
def h(L):
    
    L[0] = L[0] + 2
    print (L) 
    return L
L = [5, 17]
        ### BEGIN SOLUTION
f(g(L.copy())), h(L.copy())
        ### END SOLUTION

