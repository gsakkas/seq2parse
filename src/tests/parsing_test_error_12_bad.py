    def change_caracteres(s, i, j, t):
    res =""
    if i<j and i<len(s) and i>0 and j>0:
        res = s[:i]+t+s[j+1:]
    elif i<0:
        if j>0 and len(s)-i<j:
            res = s[:len(s)-i]+t+s[j+1:]
        elif j<0 and i>j:
            res = s[:len(s)-i]+t+s[len(s)-j-1:]
    elif j<0:
        if len(s)-j-1>i:
            res = s[:i]+t+s[len(s)-j-1:]
    return res
change_caracteres ('Comment vas-tu ?', 1, 13, '$$')
