# Write a Python program to get a list, sorted in increasing order by the last element
# in each tuple from a given list of non-empty tuples.
# Sample List : [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
# Expected Result : [(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)]

from Week2.Utilities import utility


class List5:
    lst = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
class List5:

    @staticmethod
    def last(n):
        return n[-1]

    def sort(self, lst):
        return sorted(lst, key=self.last)


lst = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
obj = List5()
result = obj.sort(lst)
print(result)
