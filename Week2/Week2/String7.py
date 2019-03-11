# 7. Write a Python program that accepts a comma separated sequence of words as input and prints the unique words
# in sorted form (alphanumerically).
# 	Sample Words : red, white, black, red, green, black
# 	Expected Result : black, green, red, white,red
from Week2.Utilities import utility


class String7:
    flag = True
    while flag:
        print("\n1.Print uniques from list in sorted order")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice == 0 or choice == 1:
                if choice == 1:
                    val = input("enter comma separated values")
                    # split values and put them in list
                    lst = val.split(',')
                    print(lst)
                    # sort in asc order
                    lst.sort()
                    # convert list into dict to remove duplicate values and back into the list
                    lstwithUniqueVals= list(dict.fromkeys(lst))

                    for uniques in lstwithUniqueVals:
                        print(uniques, end=" ")
                if choice == 0:
                    flag = False
            else:
                raise ValueError
        except ValueError:
            print("\nPlease give valid input")
            print("Try again.....")