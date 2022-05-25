def happyBirthday(person):
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print("Happy Birthday, dear " + person + ".")
    print("Happy Birthday to you!")

def main():
    running = True
    while running:
        """Enter -9 to end this loop"""
    name1 = input("Enter the BD of person name's: ")
    if name1 == '9': break
        
    else:
        happyBirthday(name1)
    
main()
