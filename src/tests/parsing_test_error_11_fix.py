phrases = ("left", "right", "left", "stop")
a = list(phrases)
c = []
for item in a:
    if item == 'right':
        c.append('left')
    else:
        c.append(item)


print(c)
