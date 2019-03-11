# 9. Write a Python program to display formatted text (width=50) as output.
from Week2.Utilities import utility


class String1:
        flag = True
        while flag:
            print("\n1.to display in formatted text")
            print("0.Exit")
            choice = int(input("Enter ur choice"))
            if choice == 1:
                try:
                    string = input("\nPlease enter your data\n")
                    CheckStr = utility.User.CheckStr1(string)
                    if CheckStr:
                        formattedData = utility.User.Format(string)
                        print("\nFormatted data in width 50 :", formattedData)
                    else:
                        raise ValueError
                except ValueError:
                    print("\nPlease enter only characters")
            if choice == 0:
                flag = False
