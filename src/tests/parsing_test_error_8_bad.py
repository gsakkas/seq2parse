Happiness = int(input("Enter your happiness on a scale of 1-100: "))
if Happiness <= 20:
        print("You lead a sad life")
elif Happiness in range(20, 50):
        print("You lead a relativity life")
elif Happiness in range(50, 75):
        print("You lead a quite happy life!")
elif Happiness in range(75, 100):
        print("You are the happiest person ever!!!")
else:
        print("Your Happiness is literally off the scales")
if Happiness in range(50, 100):
        age = int(input("Please enter your age: "))
        if age <= 12:
                print("You are still a child")
        elif age in range(13, 19):
                print("You are a teenager")
        elif age in range(20, 30):
                print("You are a young adult")
        else:
                print("You are an old adult")
        if age in range(13, 30):
                hobby = int(input("Please enter what activity makes you happiest: "))
                if hobby <= 0:
                    print("Weird")
                else
                    print("Thank you for completing thiz quiz")
