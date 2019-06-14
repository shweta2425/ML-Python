# Write a program to find the probability of drawing an ace after drawing an ace on the first draw

from Week3.ProbabilityandStatistics.Utility.utility import User


class ProbOfAceAfterAce:
    # total cards
    cards = 52
    print("\nTotal no.of cards", cards)

    # total no.of Ace in pack of cards
    ace = 4
    print("Total no.of Ace in pack of cards", ace)

    # remaining cards after drawing a Ace
    cards = cards - 1
    print("\nremaining cards after drawing a Ace", cards)

    # total remaining Ace after drawing Ace on first draw
    ace = ace - 1
    print("\nremaining Ace after drawing Ace on first draw", ace)

    # creating Utility class object
    utilityObj = User()

    def findProbability(self):
        # calling method using utility class object to find probability
        probability = self.utilityObj.calProbability(self.ace, self.cards)

        # printing probability
        print("\nProbability of drawing an ace after drawing an ace on the first draw ",
              self.ace, "/", self.cards, " =", probability)


# creating object of current class
obj = ProbOfAceAfterAce()

# calling method by using class object
obj.findProbability()
