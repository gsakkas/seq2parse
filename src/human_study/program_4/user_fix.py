def recurPower(base, exp):
    result = 1
    if exp == 1:
        return 1
    else:
        return exp * recurPower(base, exp - 1)

print(recurPower(2, 3))