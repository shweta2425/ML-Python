# 5. Write a Python function that takes a list of words and returns the length of the longest one.

from Week2.Utilities import utility


class String5:
    flag = True
    while flag:
        print("\n1.returns the length of the longest one")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice == 0 or choice == 1:
                if choice == 1:
                    size = int(input("how many words do u wanna insert in list"))
                    newList = utility.User.CreateList(size)
                    Maxlen = utility.User.FindMaxWord(newList)
                    print("list is ",newList)
                    print("the length of the longest one is",Maxlen)

                if choice == 0:
                    flag = False
            else:
                raise ValueError
        except ValueError:
            print("\nPlease give valid input")
            print("Try again.....")