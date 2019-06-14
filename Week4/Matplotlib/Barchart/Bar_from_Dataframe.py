#  Write a Python program to create bar plot from a DataFrame.
# Sample Data Frame:
# a b c d e
# 2 4,8,5,7,6
# 4 2,3,4,2,6
# 6 4,7,4,7,8
# 8 2,6,4,8,6
# 10 2,4,3,3,2

from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from Week4.Matplotlib.Utility.utility import UtilityClass


class Bar_chart_from_data_Frame:
    # creates utility class object
    utility_obj = UtilityClass()

    def create_bar(self):
        print("enter no.of list you wanna create")
        size = self.utility_obj.accept_size()

    def Create_Lists():

        flag = True
        set2 = set()
        while flag:
            ch = input("Wanna add val ?y/n")
            if ch == 'y' or ch == 'Y':
                val = input("enter value")
                set2.add(val)
            if ch == 'n' or ch == 'N':
                flag = False
        return set2

        a=np.array([[4,8,5,7,6],[2,3,4,2,6],[4,7,4,7,8],[2,6,4,8,6],[2,4,3,3,2]])
        df=DataFrame(a, columns=['a','b','c','d','e'], index=[2,4,6,8,10])

        df.plot(kind='bar')
        # Turn on the grid
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

        plt.show()

