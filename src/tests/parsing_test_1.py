def helper(a):
    if a % 3 <= 0:
        return "Good"
    else:
        return "Bad"

for i in range(12):
    print(helper(i))
