a = int(input('Zadaj cčíčíslo '))
b = int(input('Zadaj kake '))
pocet = 0
while a > 0:
    c = a % 10
    if c:
        pocet = +1
    a = a // 10
print(pocet)
