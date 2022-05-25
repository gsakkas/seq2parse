def main():

    A=[[2,3,5,1],[-1,4,7,2],[0,1,2,-4]]
    B=[[-1,4],[2,0],[-3,5],[3,1]
    [4,5,9,1]]
    results=[[0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]]

for i in range(len(X)):
   for j in range(len(Y[0])):
            for k in range (len(B)):
                result[i][j]+=A[i][k] * B[k][j]

for r in result:
        print(r)



main()
