# Write a Python program to count the number of strings where the string length is 2 or more
# and the first and last character are same from a given list of strings.
# 	Sample List : ['abc', 'xyz', 'aba', '1221']
# 	Expected Result : 2

from Week2.Utilities import utility


class List4:

    lst=['abc', 'xyz', 'aba', '1221']
    obj = utility.User.CountFirstLast(lst)
    print("Count is ", obj)

