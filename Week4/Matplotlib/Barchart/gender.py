# 10. Write a Python program to create bar plot of scores by group and gender.
# Use multiple X values on the same chart for men and women.
# Sample Data:
# Means (men) = (22, 30, 35, 35, 26)
# Means (women) = (25, 32, 30, 35, 29)


import matplotlib.pyplot as plt
import numpy as np
from Week4.Matplotlib.Utility.utility import UtilityClass


class Gender:

    # creates utility class object
    utility_obj = UtilityClass()

    def Multi_bar(self):
        # create no.of groups
        print("how many groups do u wanna create")
        n_groups = self.utility_obj.accept_size()
        # accepts men data
        print("enter men means")
        men_means = self.utility_obj.CreateList(n_groups)
        # accepts women data
        print("enter women means")
        women_means = self.utility_obj.CreateList(n_groups)

        # create plot or creates object of subplot
        fig, ax = plt.subplots()

        index = np.arange(n_groups)
        bar_width = 0.35

        # plotting men means values to create bar chart
        plt.bar(index, men_means, bar_width, color='g', label='Men')

        # plotting women means values to create bar chart
        plt.bar(index + bar_width, women_means, bar_width, color='r', label='Women')

        # Set the x axis label
        plt.xlabel('Person')

        # Set the y axis label
        plt.ylabel('Scores')

        # Sets a title
        plt.title('scores by group and gender')

        # group label
        plt.xticks(index + bar_width, ('G1', 'G2', 'G3', 'G4', 'G5'))

        # show a legend on the plot
        plt.legend()

        # Display a figure.
        plt.show()


# creates class object
obj = Gender()
# calling method by using class object
obj.Multi_bar()