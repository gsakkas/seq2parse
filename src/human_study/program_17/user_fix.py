def simple_pig_latin(input, sep="", end="."):
    sentence = ""
    for word in input.split():
        vowel = ("a", "e", "i", "o", "u")
        if word[0] in vowel:
            pig = word + "way "
            sentence.join(pig)
        else:
            new_word = word
    return pig + sentence

simple_pig_latin("i like this", sep="-", end="!")
