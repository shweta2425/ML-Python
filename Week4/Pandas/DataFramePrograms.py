import numpy as np
import pandas as pd


class DataFramePrograms:
    def __init__(self):
        self.exam_data = {
            'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin',
                     'Jonas'],
            'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
            'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
            'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

        self.labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    def print_row_col_of_DF(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        # print the rows where number of attempts in the examination is less than 2 and score greater than 15
        print("\n", df[(df['attempts'] < 2) & (df['score'] > 15)])

    def change_row_value(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        df.loc['d', 'score'] = 11.5
        print("\n after changing the value of score in row 'd' to 11.5\n\n", df)

    def sum_attempts(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print("\nthe sum of the examination attempts by the students is ", df['attempts'].sum())

    def find_mean(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        print("\nmean =", df['score'].mean())

    def add_del_row(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)

        # creates DataFrame
        data = {'name': ["Suresh"], 'score': [15.5], 'attempts': [1], 'qualify': ["yes"]}
        label = ["k"]
        df1 = pd.DataFrame(data, index=label)
        # print(df1)

        # adds specified row to DataFrame
        df = df.append(df1)
        print("\nafter adding new row\n", df)
        # deletes specified row from DataFrame
        df = df.drop('k')
        print("\n after deleting new row from DataFrame\n", df)

    def sort_DataFrame_by_name(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        # sort data frame in descending order by specified column
        df = df.sort_values(by=['name'], ascending=False)
        print("desc by name\n", df)
        # sort data frame in ascending order by specified column
        df = df.sort_values(by=['score'], ascending=True)
        print("asc by score\n", df)

    def replace_values(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        # replaces values
        df['qualify'] = df['qualify'].replace({'yes': True, 'no': False})
        print("\nDataFrame after replacing values\n", df)

    def delete_column(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        del df['attempts']
        print("\nDataFrame after deleting attempts column\n\n", df)

    def add_column(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        percentage = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        df['percentage'] = percentage
        print("\nafter adding new column\n", df)

    def iterate_row(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        # iterating over rows using iterrows() function
        for i, j in df.iterrows():
            print(i, j)
            print()

    def get_header_column_list(self):
        # creates DataFrame
        df = pd.DataFrame(self.exam_data, index=self.labels)
        print(df)
        # creates list of column headers values
        print("\nlist of column header is\n", list(df.columns.values))

    def menu(self):
        print("1.print the rows where number of attempts in the examination is less than 2 and score greater than 15.")
        print("2.change the score in row 'd' to 11.5")
        print("3.calculate the sum of the examination attempts by the students")
        print("4.calculate the mean score for each different student in DataFrame.")
        print("5.add & delete new row from DataFrame")
        print("6.sort the DataFrame first by 'name' in descending order, then by 'score' in ascending order.")
        print("7.replace the 'qualify' column contains the values 'yes' and 'no' with True and False.")
        print("8. delete the 'attempts' column from the DataFrame")
        print("9.add new column in DataFrame")
        print("10.iterate over rows in a DataFrame")
        print("11. get list from DataFrame column headers.")
        print("0.exit")
        flag = False

        while not flag:
            try:
                choice = int(input("\nEnter ur choice"))
                if not choice <= 0 and choice <= 11:

                    if choice == 1:
                        obj.print_row_col_of_DF()

                    if choice == 2:
                        obj.change_row_value()

                    if choice == 3:
                        obj.sum_attempts()

                    if choice == 4:
                        obj.find_mean()

                    if choice == 5:
                        obj.add_del_row()
                    if choice == 6:
                        obj.sort_DataFrame_by_name()

                    if choice == 7:
                        obj.replace_values()

                    if choice == 8:
                        obj.delete_column()

                    if choice == 9:
                        obj.add_column()

                    if choice == 10:
                        obj.iterate_row()

                    if choice == 11:
                        obj.get_header_column_list()

                    if choice == 0:
                        flag = True
                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input and Try again")


# creates class object
obj = DataFramePrograms()
flag = False
# calling method by using class object
obj.menu()
