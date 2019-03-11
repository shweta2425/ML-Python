# Write a Python program to split a list based on first character of word.
from Week2.Utilities import utility


class List16:

    def SplitList(self,lst):
        lst.sort()
        for char in lst:
            print(char[0].split(), "splited")
            print(char)
            print(char[0])


lst = ['zzz', 'my', 'shweta', 'abc', 'aa', 'dfdg', 'werewr']
obj = List16()
obj.SplitList(lst)

