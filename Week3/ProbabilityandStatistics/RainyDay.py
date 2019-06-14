""" In my town, it's rainy one third of the days. Given that it is rainy, there will
be heavy traffic with probability 12, and given that it is not rainy,
there will be heavy traffic with probability 14. If it's rainy and there is heavy traffic,
I arrive late for work with probability 12. On the other hand, the probability of being late is
reduced to 18 if it is not rainy and there is no heavy traffic.
In other situations (rainy and no traffic, not rainy and traffic)
the probability of being late is 0.25. You pick a random day."""


# Write a program to find following
# a.What is the probability that it's not raining and there is heavy traffic and I am not late?
# b.What is the probability that I am late?
# c.Given that I arrived late at work, what is the probability that it rained that day?


class RainyDay:

    def __init__(self):
        self.rainy = 1 / 3
        self.rainy_traffic = 1 / 2
        self.rainy_traffic_late = 1 / 2
        self.rainy_traffic_notLate = 1 / 2

        self.rainy_noTraffic = 1 / 2
        self.rainy_noTraffic_late = 1 / 4
        self.rainy_noTraffic_notLate = 3 / 4

        self.notRainy = 2 / 3
        self.notRainy_traffic = 1 / 4
        self.notRainy_traffic_late = 1 / 4
        self.notRainy_traffic_notLate = 3 / 4

        self.notRainy_notTraffic = 3 / 4
        self.notRainy_notTraffic_late = 1 / 8
        self.notRainy_notTraffic_notLate = 7 / 8

    def notRainy_Traffic_notLate(self):

        print("the total probability of it's not raining and there is heavy traffic and I am not late",
              ((self.notRainy) * (self.notRainy_traffic) * (self.notRainy_traffic_notLate)))

    def late(self):
        print("if Rainy,there is a traffic and i am late then probability=",
              self.rainy * self.rainy_traffic * self.rainy_traffic_late)
        print("if Rainy,there is no traffic and i am late then probability=",
              self.rainy * self.rainy_noTraffic * self.rainy_noTraffic_late)
        print("if not Rainy,there is a traffic and i am late then probability=",
              self.notRainy * self.notRainy_traffic * self.notRainy_traffic_late)
        print("if not Rainy,there is a no traffic and i am late then probability=",
              self.notRainy * self.notRainy_notTraffic * self.notRainy_notTraffic_late)
        print("probability that I am late=",self.rainy * self.rainy_traffic * self.rainy_traffic_late+
              self.rainy * self.rainy_noTraffic * self.rainy_noTraffic_late+
              self.notRainy * self.notRainy_traffic * self.notRainy_traffic_late+
              self.notRainy * self.notRainy_notTraffic * self.notRainy_notTraffic_late)

    def late_rainy(self):

        print("I am late,its rainy,heavy traffic then probability=",
              self.rainy*self.rainy_traffic*self.rainy_traffic_late)
        print("I am late,its rainy,there is no traffic then probability=",
              self.rainy * self.rainy_noTraffic * self.rainy_noTraffic_late)
        print("I arrived late at work, what is the probability that it rained that day",
              self.rainy*self.rainy_traffic*self.rainy_traffic_late +
              self.rainy * self.rainy_noTraffic * self.rainy_noTraffic_late)

    def menu(self):
        print("1.the probability that it's not raining and there is heavy traffic and I am not late")
        print("2.the probability that I am late")
        print("3.I arrived late at work, what is the probability that it rained that day")
        print("0.exit")
        flag = False

        while flag == False:
            try:
                choice = int(input("Enter ur choice"))
                if choice >= 0 and choice <= 3:
                    if choice == 1:
                        obj.notRainy_Traffic_notLate()
                    if choice == 2:
                        obj.late()
                    if choice == 3:
                        obj.late_rainy()
                    if choice == 0:
                        flag = True

                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input")
                print("Try again.....")


obj = RainyDay()
flag = False
# calling method by using class object
obj.menu()

