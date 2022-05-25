def ct1(s):
    print(chr(ord('G') + ord(s[1]) -ord(s[0])), end='')
print(chr(ord('G') + ord(s[1])
-
ord(s[0])), end='')
t, count = '', 0
for c in s:
        if (not c.isalnum()): t += c
if (c.isdigit()): print(c, end='')
elif (c.isupper()): print(c.lower(), end='')
else: count += 1
return
\
\
print(ct1('ae1#B2cD!'))
