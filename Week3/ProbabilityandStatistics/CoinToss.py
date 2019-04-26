# You toss a fair coin three times write a program to find following:
# a. What is the probability of three heads, HHH?
# b.What is the probability that you observe exactly one heads?
# c.Given that you have observed at least one heads, what is the probability that you observe at least two heads?
from Week3.ProbabilityandStatistics.Utility.utility import User


class CoinToss:
    # if coin toss 3 times then combination of outcome can be as follow
    SampleList = ['HHH', 'HHT', 'HTH', 'THH', 'TTT', 'TTH', 'THT', 'HTT']
    # returns length of a list
    count = len(SampleList)
    print("\nPossibilities if toss a fair coin three times", SampleList)
    print("\nTotal possibilities of outcome are", count)

    # creating Utility class object
    utilityObj = User()

    def findProbability(self):
        # counts total no.of specified item in a list
        possibilities = self.SampleList.count('HHH')
        print("the possibilities of three heads (HHH) are ", possibilities)

        # calling method using utility class object to find probability
        probability = self.utilityObj.calProbability(possibilities, self.count)

        # printing the probability
        print("\nProbability of three heads is ", possibilities, "/", self.count, "=", probability)

    def FindOneHead(self):
        oneHead = self.utilityObj.oneHead(self.SampleList)
        print("\nPossibilities of you observe exactly one head are", oneHead)

        # calling method using utility class object to find probability
        probability = self.utilityObj.calProbability(oneHead, self.count)
        print("\nprobability that you observe exactly one heads is", oneHead, "/", self.count, "=", probability)

    def FindTwoHead(self):

        atleastoneHead = self.utilityObj.AtleastOneHead(self.SampleList)
        print("\nPossibilities of you observe at least one head is", atleastoneHead)

        # calling method using utility class object to find probability
        probabilityA1 = self.utilityObj.calProbability(atleastoneHead, self.count)
        print("\nprobability that you observe at least one heads is", atleastoneHead, "/", self.count, "=", probabilityA1)


        twoHead = self.utilityObj.FindTwoHead(self.SampleList)
        print("\nPossibilities of you observe at least two heads are", twoHead)

        # calling method using utility class object to find probability
        probabilityA2 = self.utilityObj.calProbability(twoHead, self.count)
        print("\nprobability that you observe at least two heads is", twoHead, "/", self.count, "=", probabilityA2)

        print("probability of A2/A1 =", (probabilityA2/probabilityA1))

    def menu(self):
        print("\n1.the probability of three heads")
        print("2.the probability that you observe exactly one heads")
        print("3.the probability that you observe at least two heads")
        print("0.exit")
        flag = False

        while flag == False:
            try:
                choice = int(input("Enter ur choice"))
                if choice >= 0 and choice <= 3:
                    if choice == 1:
                        obj.findProbability()
                        flag = True
                        if flag == True:
                            self.menu()
                    if choice == 2:
                        obj.FindOneHead()
                    if choice == 3:
                        obj.FindTwoHead()
                    if choice == 0:
                        flag = True

                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input")
                print("Try again.....")


# creating object of current class
obj = CoinToss()
flag = False
# calling method by using class object
obj.menu()
