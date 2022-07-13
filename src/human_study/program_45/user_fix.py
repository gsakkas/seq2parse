import random
total = 90

while total > 85 and total < 150:
    player_choice = int(input("Enter 1 for heads and 0 for tails: "))
    if player_choice == random.randint(0, 1):
        total += 9
        print("Your total is {}.".format(total))
    else:
        total -= 10
        print("Your total is {}.".format(total))
