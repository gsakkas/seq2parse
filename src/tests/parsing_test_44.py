print("hello")


num_hour = 10
num_min = 40
#rush is 5:30am - 9:30am
if (num_hour >= 5 and num_min >= 30
	or num_hour <= 9 and num_min <= 30):
	print("Rush")
else:
    print("nahh")

