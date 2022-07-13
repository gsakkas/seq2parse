class Buf:
    list = []

    def add(self, *a):
        self = a

s = Buf()
print(s.add(1, 2, 3))
