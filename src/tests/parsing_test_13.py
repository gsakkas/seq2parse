import math
def main():
    x = int(input())
    factorize(x)

def factorize(x):
    print(x, 'is ', end=' ')
    ie = math.sqrt(x)
    ie = int(ie)
    for i in range(2, ie):
        if (x % i == 0):
            while (x > 0):
                if (x % i == 0):
                    print(i, end=' ')
                x /= i
                if i > ie:
                    break
                    print('*', end=' ')
                else:
                    i += 1
    else:
        print('pirme')

if __name__ == '__main__':
    main()
