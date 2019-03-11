# Write a Python program to remove duplicates from a list of lists.
# 	Sample list : [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
# 	New List : [[10, 20], [30, 56, 25], [33], [40]]

from Week2.Utilities import utility


class List15:
    lst = [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
    print("List is :", lst)

    obj = utility.User.RemoveDupl(lst)
    print("List without duplicates :", obj)
