# 1. Write a Python program to calculate the length of a string.

from Week2.Utilities import utility


class String1:
        flag = True
        while flag:
            print("\n1.Calculate length of str")
            print("0.Exit")
            try:
                choice = int(input("Enter ur choice"))
                if choice == 0 or choice == 1:
                    if choice == 1:
                        try:
                            string = input("\nPlease enter your name")
                            CheckStr = utility.User.CheckString(string)
                            if CheckStr:
                                length = utility.User.CheckLen(string)
                                print("\nLength of the string", string, "is :", length)
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
