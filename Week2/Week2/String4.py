# 4. Write a Python program to add 'ing' at the end of a given string (length should be at least 3).
# If the given string already ends with 'ing' then add 'ly' instead. If the string length of the given
# string is less than 3, leave it unchanged.
# 	Sample String : 'abc'
# 	Expected Result : 'abcing'
# 	Sample String : 'string'
# 	Expected Result : 'stringly'

from Week2.Utilities import utility


class String4:
    flag = True
    while flag:
        print("\n1.Add ing or ly in a str")
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
                            newStr = utility.User.AddIng(str1)
                            print(newStr)
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
