# 3. Write a Python program to get a string from a given string where all occurrences of its first char have been
# changed to '$', except the first char itself.
# 	Sample String : 'restart'
# 	Expected Result : 'resta$t'

from Week2.Utilities import utility


class String3:
    flag = True
    while flag:
        print("\n1.Replace Occurrence with $ in a str")
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
                            newStr = utility.User.ReplaceStr(str1)
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
