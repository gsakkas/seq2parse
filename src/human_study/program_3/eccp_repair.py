minimum_num = int(input("Enter the minimum number for the range"))
maximum_num = int(input("Enter the Maximum number for the range"))
num_1 = int(int(input("Enter a Number between the range")))

if num_1 in range(minimum_num, maximum_num):
    print("The range is", num_1)()
else:
    print("The number is not in the Range")