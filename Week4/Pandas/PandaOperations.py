import numpy as np
import pandas as pd

from Week4.Pandas.Utility.utility import User


class PandaOperations:
    def __init__(self):
        self.arr = np.array([1, 2, 3, 4, 5])
        self.exam_data = {
            'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin',
                     'Jonas'],
            'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
            'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
            'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

        self.labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    utility_obj = User()

    def create_series(self):
        # creates series (1-d array)
        series = self.utility_obj.creates_series(self.arr)
        print(series)

    def convert_to_list(self):
        # creates series (1-d array)
        series = self.utility_obj.creates_series(self.arr)
        print(series)
        print(type(series))
        # converts into list
        list1 = series.tolist()
        # prints type of list
        print(list1, type(list1))

    def series_arithmetic(self):
        series1 = pd.Series([2, 4, 6, 8, 10])
        series2 = pd.Series([1, 3, 5, 7, 9])

        print("series1 =\n", series1)
        print("series2 =\n", series2)

        add = series1 + series2
        print("Addition two Series:")
        print(add)

        print("Subtraction two Series:")
        sub = series1 - series2
        print(sub)
        print("Multiplication two Series:")
        mul = series1 * series2
        print(mul)
        print("Divide Series1 by Series2:")
        div = series1 / series2
        print(div)

    def power(self):
        arr = np.arange(0, 7)
        print("array is", arr)
        # return power of elements
        print(np.power(arr, 3))

    def create_dataframe(self):
        # creates data frame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)

    def display_summary(self):
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print("summary of given DataFrame\n")
        # print summary of information of data frame
        print(df.info())

    def print_rows_of_dataframe(self):
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print("\nfirst 3 rows of DataFrame are\n ")
        # row slicing
        print(df[:3])

    def print_column_of_dataframe(self):
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print("\nname column\n")
        print(df['name'])

        print("\nscore column\n")
        print(df['score'])

    def print_row_column(self):
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        # print specified rows and columns
        print("\n",df.iloc[[1, 3, 5, 6], [0, 1]])

    def print_attempt(self):
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print("\nthe rows where the number of attempts in the examination is greater than 2.")
        # print data frame rows where column attempts's value is greater than 2
        print(df[df['attempts']>2])
    
    def count_row_column(self):
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print("no.of rows & columns")
        # return no.of rows & columns of data frame
        print(df.shape)
        # return length of index or row of data frame
        print("no.of rows are",len(df.axes[0]))
        # return length of column of data frame
        print("no.of columns are", len(df.axes[1]))

    def print_empty_score_row(self):
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print(" the rows where the score is missing, i.e. is NaN. ")
        # check if specified column values are empty
        print(df[df['score'].isnull()])

    def menu(self):
        print("1.create series from ndarray")
        print("2.convert a Panda module Series to Python list and it's type.")
        print("3.arithmetic operations with series")
        print("4.print powers of an array values element-wise")
        print("5.create dataFrame from given dictionary")
        print("6.display a summary of the basic information about a specified Data Frame and its data")
        print("7.print first 3 rows of a given DataFrame")
        print("8.select the 'name' and 'score' columns from the given DataFrame.")
        print("9.print the specified columns and rows from a given data frame")
        print("10.print the rows where the number of attempts in the examination is greater than 2. ")
        print("11.count the number of rows and columns of a DataFrame")
        print("12.print the rows where the score is missing, i.e. is NaN")
        print("0.exit")
        flag = False

        while not flag:
            try:
                choice = int(input("\nEnter ur choice"))
                if not choice <= 0 and choice <= 15:

                    if choice == 1:
                        obj.create_series()

                    if choice == 2:
                        obj.convert_to_list()

                    if choice == 3:
                        obj.series_arithmetic()

                    if choice == 4:
                        obj.power()

                    if choice == 5:
                        obj.create_dataframe()

                    if choice == 6:
                        obj.display_summary()

                    if choice == 7:
                        obj.print_rows_of_dataframe()

                    if choice == 8:
                        obj.print_column_of_dataframe()

                    if choice == 9:
                        obj.print_row_column()

                    if choice == 10:
                        obj.print_attempt()

                    if choice == 11:
                        obj.count_row_column()

                    if choice == 12:
                        obj.print_empty_score_row()

                    if choice == 0:
                        flag = True
                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input and Try again")


# creates class object
obj = PandaOperations()
flag = False
# calling method by using class object
obj.menu()


