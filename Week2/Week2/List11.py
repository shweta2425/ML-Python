# Write a Python program to generate all permutations of a list in Python.
from Week2.Utilities import utility


class List11:
    lst = [1, 2, 3]
    obj = utility.User.FindCombi(lst)
    print("Combinations are :")
    for i in obj:
        print(i)
