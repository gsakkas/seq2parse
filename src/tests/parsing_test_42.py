import random

class Card(Object):

    def __init__(self, color, value):
                self.color = color
                self.value = value

    def show(self):
        print ("{} of {}".format(self.value, self.color))

class Deck(object):

    def __init__(self):
        self.cards = []
        self.build()

    def build(self):
        for s in [Spades, Clubs, Diamonds, Hearts]:
            for v in range(1,14):
                self.cards.append(Card(s,v))

    def show(self):
        for c in self.cards:
            c.show()

    def shuffle(self):
        for i in range(len(self.cards)-1,0,-1):
            r = random.randint(0,i)
            self.cards[i], self.card[r] = self.cards[r]. self.cards[i]

    def draw(self):
        return self.cards.pop()

class Player(object):
    def __init__(self, name):
        self.name = name
        self.hand = []

    def draw(self,deck):
        self.hand.append(deck.drawcard())

    def showHand(self):
        for card in self.hand:
            card.show()

    def discard(self):
        return self.hand.pop()
