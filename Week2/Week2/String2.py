# 2. Write a Python program to count the number of characters (character frequency) in a string.
# 	Sample String : google.com
# 	Expected Result : {'o': 3, 'g': 2, '.': 1, 'e': 1, 'l': 1, 'm': 1, 'c': 1}

from Week2.Utilities import utility


class String2:
    flag = True
    while flag:
        print("\n1.Calculate number of characters in a str")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice == 0 or choice == 1:
                if choice == 1:
                    try:
                        string = input("\nPlease enter a string")
                        strtype = utility.User.CheckStr1(string)
                        if strtype:
                            str1 = str(string)
                            newDict = utility.User.CreateDict(str1)
                            print(newDict)
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
