x =  [1,2,3,4,5]
def foo(lst):
    return [i * lst[i] for i in range(len(lst)) if i %2 == 0]
    
print(foo(x))
