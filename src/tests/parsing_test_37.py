""" Enter docstring for this program
"""
import random
import matplotlib.pyplot as pyplot

def guessingGame1(limit):
    """Play the guessing game by making random guesses."""

    secretNumber = random.randrange(1, limit + 1)
    myGuess = 0
    guesses = 0
    while (myGuess != secretNumber):
        myGuess = random.randrange(1, limit + 1)
        guesses = guesses + 1

    return guesses

def monteCarlo(limit, trials):

    totalGuesses_game1=0

    for sequence in range(trials):
        guessesNum_g1= guessingGame1(limit)
        totalGuesses_game1= totalGuesses_game1+ guessesNum_g1
    return totalGuesses_game1/trials


def main():
    """Enter docstring for main function
    """
    trials = int(input("Enter number of trials for simulation run: "))
    limit = int(input("enter max number for range: "))

    monteCarlo(limit, trials)

main()
