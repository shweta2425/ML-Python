# Given the following statistics, write a program to find the probability that a woman
# has cancer if she has a positive mammogram result?
# 	a. One percent of women over 50 have breast cancer.
# 	b. Ninety percent of women who have breast cancer test positive on mammograms.
# 	c. Eight percent of women will have false positives.


class Mammogram:
    def __init__(self):
        # 1%women have cancer
        self.probA = 0.01
        # 99% women don't have cancer
        self.probComplimentoryA = 0.99
        # Ninety percent of women who have breast cancer test positive on mammograms
        self.XandA = 0.90
        # Eight percent of women will have false positives.
        self.XlcomplimentoryA = 0.08

    def calProbability(self):
        print("We assume ;\n Event A = woman has cancer\n X=she has positive Mammogram")
        print("One percent of women over 50 have breast cancer i.e P(A)=", self.probA)
        print(self.probComplimentoryA, "don't have cancer i.e P(~A)=", self.probComplimentoryA)
        print("Ninety percent of women who have breast cancer test positive on mammograms i.e P(X|A)=", self.XandA)
        print("Eight percent of women will have false positives i.e P(X|~A)=", self.XlcomplimentoryA)

        probability = ((self.XandA * self.probA) /
                        ((self.XandA * self.probA) + (self.XlcomplimentoryA * self.probComplimentoryA)))
        print("Probability =", probability)


obj = Mammogram()
obj.calProbability()
