class Rect:
    def __init__(self, l, b):
        self.length = l
        self.breadth = b
    def area(self):
        return self.length * self.breadth
# initialize a Rect object r1 with length 20 and breadth 10
r1 = Rect(20, 10)
r2 = Rect(40, 30)
print('area of r1 : ', r1.area()) # Rect.area(r1)
print('area of r2 : ', r2.area()) # Rect.area(r2)
