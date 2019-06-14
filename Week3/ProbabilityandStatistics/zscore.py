# X is a normally normally distributed variable with mean μ = 30 and standard deviation σ = 4. Write a program to find
# a. P(x < 40)
# b. P(x > 21)
# c. P(30 < x < 35)


class Zscore:
    def __init__(self):
        self.mean = 30
        self.num = 40
        self.deviation = 4
        self.ZtableVal1 = 0.9938
        self.X = 21
        self.Xztableval = 0.0122
        self.X2 = 30
        self.X3 = 35
        self.X2ztableval = 0.5
        self.X3ztableval = 0.8944

    def findzscore(self):
        zscore = (self.num - self.mean) / self.deviation
        print("zscore =", zscore)
        obj.findZtableVal(zscore)

    def findZtableVal(self, zscore):
        print("area to the left of", zscore, "=", self.ZtableVal1)

    def findzscoreX(self):
        zscore = (self.X - self.mean) / self.deviation
        print("zscore =", zscore)
        obj.findZtablevalX(zscore)

    def findZtablevalX(self, zscore):
        print("area to the left of", zscore, "=", self.Xztableval)
        print("Hence P(x > 21) =", 1 - self.Xztableval)

    def findzscoreX2X3(self):
        zscoreX2 = (self.X2 - self.mean) / self.deviation
        zscoreX3 = (self.X3 - self.mean) / self.deviation
        print("zscore of =", self.X2, "=", zscoreX2)
        print("zscore of =", self.X3, "=", zscoreX3)
        obj.findZtablevalX2X3(zscoreX2, zscoreX3)

    def findZtablevalX2X3(self, zscoreX2, zscoreX3):
        print("area to the left of", zscoreX2, "=", self.X2ztableval)
        print("area to the left of", zscoreX3, "=", self.X3ztableval)

        print("P(30 < x < 35) =", self.X3ztableval - self.X2ztableval)

    def menu(self):
        flag = False
        print("P(x < 40)")
        print("P(x > 21)")
        print("P(30 < x < 35)")

        while flag == False:
            try:
                choice = int(input("\nEnter ur choice"))
                if choice >= 0 and choice <= 3:

                    if choice == 1:
                        obj.findzscore()

                    if choice == 2:
                        obj.findzscoreX()
                    if choice == 3:
                        obj.findzscoreX2X3()
                    if choice == 0:
                        flag = True

                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input")
                print("Try again.....")


obj = Zscore()
flag = False
# calling method by using class object
obj.menu()
