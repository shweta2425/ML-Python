# 8. Write a Python program to get the last part of a string before a specified character.
# 	https://www.w3resource.com/python-exercises
# 	https://www.w3resource.com/python

# txt = "apple#banana#cherry#orange"
# # setting the max parameter to 1, will return a list with 2 elements!
# x1 = (txt.rsplit("#", 1)[1])
# print(x1)

from Week2.Utilities import utility


class String1:
        flag = True
        while flag:
            print("\n1.to get the last part of a string before a specified character of str")
            print("0.Exit")
            try:
                choice = int(input("Enter ur choice"))
                if choice == 0 or choice ==1:
                    if choice == 1:
                        try:
                            string = input("\nPlease enter your string")
                            CheckStr = utility.User.CheckString(string)
                            if CheckStr:
                                Splitedstr = utility.User.Rsplit(string)
                                print(Splitedstr[1],Splitedstr[0])
                            else:
                                raise ValueError
                        except ValueError:
                            print("\nPlease enter only characters")

                    if choice == 0:
                        print("\nThank You !!!!!!!")
                        flag = False
                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input")
                print("Try again.....")