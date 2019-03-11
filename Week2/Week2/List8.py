# Write a Python program to find the list of words that are longer than n from a given list of words
from Week2.Utilities import utility


class List8:

    lst=["I am Learning Python ML and it's very interesting"]
    size = int(input("Enter size"))
    obj = utility.User.CalLength(size, lst)
    print(obj)


