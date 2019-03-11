# 11. Write a Python program to reverse a string.
from Week2.Utilities import utility


class String11:
        flag = True
        while flag:
            print("\n1.Reverse a str")
            print("0.Exit")
            try:
                choice = int(input("Enter ur choice"))
                if choice == 0 or choice == 1:
                    if choice == 1:
                        try:
                            string = input("\nPlease enter your str")
                            CheckStr = utility.User.CheckStr1(string)
                            if CheckStr:
                                reverseStr = utility.User.ReverseStr(string)
                                print("\nOriginal string", string)
                                print("\nReverse string", reverseStr)

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