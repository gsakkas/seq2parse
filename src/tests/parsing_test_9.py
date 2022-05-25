s = input().strip().lower()
t = input().strip().lower()
n = 0
m = len(s)
while n < m and n+len(t)<= m:  
    if t in s[n:m]:
        a = s[n:m].find(t)
        print(s[n:m])
        print(s[a-1],s[a+len(t)])
        if (a == 0 and s[n:m][a+len(t)] in """ "',.""") or\
        s[a-1] in """ "',.""" and s[n:m][a+len(t)] in """ "',.""" :
              print("Found")
              break
        else:
            print("Not Found")
            break
        n = a+len(t)+1
    else:
        print("Not Found")
        break
