# 10. Write a Python program to count occurrences of a substring in a string.
from Week2.Utilities import utility


class String10:
    flag = True
    while flag:
        print("\n1.Count occurrences of a substring in a string")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice == 0 or choice == 1:
                if choice == 1:
                    try:
                        string = input("\nPlease enter your str")
                        CheckStr = utility.User.CheckString(string)
                        print(CheckStr)
                        substr = input("\n enter the substring u wanna search in the string")
                        CheckSubStr = utility.User.CheckStrsubstr(string,substr)

                        if CheckStr==True and CheckSubStr == True:
                            occurrence = utility.User.CountOccurrence(string, substr)
                            print("\nOccurrence of", substr, "in", string, "is", occurrence)

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