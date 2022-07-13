def simetrica(m):
    if len(m) == len(m[0]):
        for i in range(len(m)):
            for j in range(len(m[i])):
                if m[i][j] != m[j][i]:
                    return False
        return True
    return False

if __name__ == "__main__":
    [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
