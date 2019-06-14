# Write a program to find the probability of drawing an ace after drawing a king on the first draw
from Week3.ProbabilityandStatistics.Utility.utility import User


class ProbOfAceAfterKing:

    # total cards
    cards = 52
    print("\nTotal no.of cards", cards)

    # total no.of Ace in pack of cards
    ace = 4
    print("Total no.of Ace in pack of cards", ace)

    # remaining cards after drawing a king
    cards = cards - 1
    print("\nremaining cards after drawing a king", cards)

    # creating Utility class object
    utilityObj = User()

    def findProbability(self):

        # calling method using utility class object to find probability
        probability = self.utilityObj.calProbability(self.ace, self.cards)

        # printing probability
        print("\nProbability of drawing an ace after drawing a king on the first draw ",
              self.ace, "/", self.cards, " =", probability)


# creating object of current class
obj = ProbOfAceAfterKing()

# calling method by using class object
obj.findProbability()
