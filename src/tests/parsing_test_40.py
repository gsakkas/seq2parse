def countMembers(s):
    c=0
    extro=['e','f','g','h','i','j','F','G','H','I','J','K','L','M','N','O','P','Q','R',',S','T','U','V','W','X','2','3','4','5','6','!',',','\\']
    for i in s:
        if s in extro:
            c=c+1
    return c
countMembers("2aAb3?eE'_13")
