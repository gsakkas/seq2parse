def etop_2(word, c):
    if c > 8:
        return etop_1(word)
    else:
        Preservatives = ["(", '"', ':', ',', '!', '.', '?', ';', ")"]
        if word[0] in Preservatives:
            xord = word[1:]
            yord = etop_2(xord, (c + 1))
            zord = word[0] + yord
        elif word[(len(word) - 1)] in Preservatives:
            xord = word[:(len(word) - 1)]
            yord = etop_2(xord, (c + 1))
            zord = yord + word[(len(word) - 1)]
        else:
            zord = etop_2(word, (c + 1))
        return zord

def etop_3(s):
    s_split = s.split(sep='\n')         #split at linebreaks
    t_split = []                        #initialize target list
    for s1 in s_split:                  #for every item after a new linebreak
        s1_split = s1.split(sep=' ')    #split at spaces
        t1_split = []                   #initialize
        for s2 in s1_split:             #for every item after a new space
            s2_split = s2.split(sep='-')#split at '-'
            t2_split = []               #initialize
            for word in s2_split:       #for every word
                word = etop_2(word, 0)  #translate
                t2_split.append(word)   #put word into t2 list
            t2 = '-'.join(t2_split)     #t2 = word-word from t2 lsit
            t1_split.append(t2)         #add t2 to t1 list
        t1 = ' '.join(t1_split)         #t1 = w-w w-w w-w-w from t1 list
        t_split.append(t1)              #add t1 to t list
    t = '\n'.join(t_split)              #t = w-w w-w\nw w\nw-w-w\nw
    return t

    def main():
        v1 = """
        CHAPTER I

Down the Rabbit-Hole

Alice was beginning to get very tired of sitting by her sister
on the bank, and of having nothing to do: once or twice she had
peeped into the book her sister was reading, but it had no
pictures or conversations in it, "and what is the use of a book,"
thought Alice "without pictures or conversation?"

So she was considering in her own mind (as well as she could,
for the hot day made her feel very sleepy and stupid), whether
the pleasure of making a daisy-chain would be worth the trouble
of getting up and picking the daisies, when suddenly a White
Rabbit with pink eyes ran close by her.

There was nothing so VERY remarkable in that; nor did Alice
think it so VERY much out of the way to hear the Rabbit say to
itself, "Oh dear! Oh dear! I shall be late!" (when she thought
it over afterwards, it occurred to her that she ought to have
wondered at this, but at the time it all seemed quite natural);
but when the Rabbit actually TOOK A WATCH OUT OF ITS
WAISTCOAT-POCKET, and looked at it, and then hurried on, Alice
started to her feet, for it flashed across her mind that she had never
before seen a rabbit with either a waistcoat-pocket, or a watch to
take out of it, and burning with curiosity, she ran across the
field after it, and fortunately was just in time to see it pop
down a large rabbit-hole under the hedge.

There was not a moment to be lost:
away went Alice like the wind, and was just in time to hear it
say, as it turned a corner, "Oh my ears and whiskers, how late
it's getting!"

Suddenly she came upon a little three-legged table, all made of
solid glass; there was nothing on it except a tiny golden key,
and Alice's first thought was that it might belong to one of the
doors of the hall; but, alas! either the locks were too large, or
the key was too small, but at any rate it would not open any of
them.
"""
        v2 = etop_3(v1)
        print()
        print(v2)

if __name__ == "__main__":
    main()
