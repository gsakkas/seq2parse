def log_base_2(number):
    x = 1
    while (number - (2x) > 0):
        x = x + 1
    target = x
    if (number - (2x) == 0):
        print(target)
    else:
        lower = x - 1
        upper = x
        print("Between %d and %d" % lower, upper)

log_base_2(256)
log_base_2(81)
