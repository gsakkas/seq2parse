s = input()
i = 0
j = 1
c =0
m =""

if s[i] != s[j]:
    while i<j and j < (len(s)):
        if s [i] != s [j]:
         m = m +s[i]+str(c+1)
         i +=1
         j +=1
         c = 0
print (m + s[i]+str(c+1))
break
while i<j and j < (len(s)-1):
    if s[i] ==s[j]:
        c +=1
        i +=1
        j +=1
    if s [i] != s [j]:
        m = m+s[i]+str(c+1)
        i +=1
        j +=1
        c = 0
print (m+s[i]+str(c+2))

