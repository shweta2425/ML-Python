# Write a Python program to get the number of occurrences of a specified element in an array.

from Week2.Utilities import utility


class Array3:
    # creating class obj
    obj = utility.User()
    # Accepts array from user
    arr1 = obj.accepts()

    def Count(self):
        num = int(input("enter ele to count"))
        cnt = self.arr1.count(num)
        print("\n Occurrence of", num, "is :", cnt)


obj = Array3()
obj.Count()
