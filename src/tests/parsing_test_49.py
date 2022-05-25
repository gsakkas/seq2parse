guesses = []
for i in range(6):
    previous_letter = ''
    letter_chosen = input('Enter a letter: ')
    if letter_chosen.isalpha():
        guesses.append(letter_chosen.upper())
        print('Here are the letters you have already guessed: ', guesses)
    elif letter_chosen.lower() == previous_letter.lower():
        print('Sorry, you have already guessed that.')
    else:
        print('Sorry', letter_chosen, 'is not a valid guess.')
    previous_letter = letter_chosen\

main()
