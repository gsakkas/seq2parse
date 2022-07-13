fin = ['shoe', 'cold', 'schooled']

def interlock(file):
    for word in file:
        for next_word in file:
            print(word[:: 1] + next_word[:: 2])
    return False

print(interlock(fin))
