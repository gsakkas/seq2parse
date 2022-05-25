def a(A):
    F = []
    i = len(A)
    while i > -1:
        F.append(A[i])
        i = i - 1
    return F
a(['apples', 'eat', "don't", 'I', 'but', 'Grapes', 'Love', 'I'])
