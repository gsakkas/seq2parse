str = input()
dub = 0
for i in range(len(str)):
    x = [i]
    if x == [i+1]:
        dub = dub + 1
if dub == 0:
    print("no double letters")
else:
    print("cotains double letters")
