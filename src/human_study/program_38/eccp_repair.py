def eCount(s):
    count = 0
    e = "Ee"
    for c in s:
        if c in e:
            count = count + 1
        return count

print(eCount(Write, sentence))
