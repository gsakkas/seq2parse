word = input()
vowels = 0
cons = 0
for i in word:
    if i == "y":
        (vowels = vowels + 1) and (cons = cons + 1)
    elif i == "a" or i == "e" or i == "i" or i == "o" or i == "u":
        vowels = vowels + 1
    elif i != "a" and i != "e" and i != "i" and i != "o" and i != "u":
        cons = cons + 1
print(str(vowels) + " vowels")
print(str(cons) + " consenants")
