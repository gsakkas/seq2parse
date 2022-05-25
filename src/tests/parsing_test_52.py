grid = [['1', '2', '3'], ['4', '5', '6'], ['.', '8', '9']]
i = 0
j = 0
while grid[i][j] != '.':
    if j < len(grid[i]):
        j = j + 1
    else:
        i = i + 1
        j = 0
pos = (i, j)
print(pos)
