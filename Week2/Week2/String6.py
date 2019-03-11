# 6. Write a Python script that takes input from the user and displays that input back in upper and lower cases.

from Week2.Utilities import utility


class String1:
        flag = True
        while flag:
            print("\n1.Convert str in upper and lower cases")
            print("0.Exit")
            try:
                choice = int(input("Enter ur choice"))
                if choice == 0 or choice == 1:
                    if choice == 1:
                        try:
                            string = input("\nPlease enter string")
                            CheckStr = utility.User.CheckStr1(string)
                            if CheckStr:
                                upper, lower = utility.User.ConvertStr(string)
                                print("\nthe string in upper case", upper)
                                print("\nthe string in lower case", lower)
                            else:
                                raise ValueError
                        except ValueError:
                            print("\nPlease enter only characters")
                    if choice == 0:
                        flag = False
                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input")
                print("Try again.....")
