# Write a program to find probability of drawing an ace from pack of cards
from Week3.ProbabilityandStatistics.Utility.utility import User


class FindAce:

    # total cards
    cards = 52
    print("\nTotal no.of cards", cards)

    # total no.of Ace in pack of cards
    ace = 4
    print("Total no.of Ace in pack of cards", ace)

    # creating Utility class object
    utilityObj = User()

    def findProbability(self):

        # calling method using utility class object to find probability
        probability = self.utilityObj.calProbability(self.ace, self.cards)

        # printing probability
        print("\nProbability of drawing an ace from pack of cards ",self.ace, "/", self.cards, "", probability)


# creating object of current class
obj = FindAce()

# calling method by using class object
obj.findProbability()
