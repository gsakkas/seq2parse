names = ["Bob", "Mark", "Jack", "Luise", "John",\
    "Amily", "Mary", "Rose"]

do_not_come ="Luise"
names.remove(do_not_come)
for name in names:
        print(name\
        + " I would like to invite you to have dinner with me.\n")

print(do_not_come + " can't show up.\n")

names[2] = "Harbor"
names.insert(0, "Bill")
names.insert(4, "Bean")
names.append("Edwerd")
name_count = len(names)
name_count = len(names)
while name_count > 2:
        name_remove = names.pop()
        print("Sorry, " + name_remove + "there's no more place. "\
                "I'll invite you next time.")
        name_count -= 1
for name in names:
        print(name + " I would like to invite you to have dinner with me.\n")

print(names)
