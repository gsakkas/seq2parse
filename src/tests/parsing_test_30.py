def binarySearchValues(L, v):

    def search(L, v, low,high):
        length = len(L)
        checkpoint = int((high - low)/2) - 1
        if v == L[checkpoint]: return [(checkpoint, L[checkpoint])]
        else:
            if ord(v) < ord(L[checkpoint]):
                return [(checkpoint, L[checkpoint])] + search(L,v,0,checkpoint)
            elif ord(v) > ord(L[checkpoint]):
                return  [(checkpoint, L[checkpoint])] +\
                search(L,v,checkpoint, length)

    return search(L,v,0, len(L))

def testBinarySearchValues():
    print('Testing binarySearchValues()...', end='')
    L = ['a', 'c', 'f', 'g', 'm', 'q']
    print((binarySearchValues(L, 'a')))

testBinarySearchValues()

