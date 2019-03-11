# 12. Write a Python program to lowercase first n characters in a string.


from Week2.Utilities import utility


class String1:
    flag = True
    while flag:
        print("\n1.lowercase first n characters in a string")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice == 0 or choice == 1:
                if choice == 1:
                    try:
                        string = input("\nPlease enter your str")
                        numofChars = int(input("first how many characters u wanna convert into lower case"))
                        CheckStr = utility.User.CheckStr1(string)
                        if CheckStr:
                            lowercaseStr = utility.User.ConvertfirstNtoLower(string, numofChars)
                            print("\nOriginal string", string)
                            print("\nSub string in lowercase", lowercaseStr)

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
