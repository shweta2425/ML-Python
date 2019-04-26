# A radar unit is used to measure speeds of cars on a motorway. The speeds are normally distributed with a mean of 90
# km/hr and a standard deviation of 10 km/hr. Write a program to find the probability that a car picked
# at random is travelling at more than 100 km/hr?

class Radar:
    def __init__(self):
        self.X = 100
        self.mean = 90
        self.deviation = 10
        self.ZtableVal = 0.8413

    def findzscore(self):
        zscore = (self.X - self.mean) / self.deviation
        print("zscore =", zscore)
        obj.findZtableVal(zscore)

    def findZtableVal(self, zscore):
        print("area to the left of", zscore, "=", self.ZtableVal)
        print("The probability that a car selected at a random has a speed greater than 100 km/hr is equal to",
              1 - self.ZtableVal)


obj = Radar()
obj.findzscore()
