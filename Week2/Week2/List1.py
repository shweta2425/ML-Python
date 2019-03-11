# Write a Python program to sum all the items in a list.
class List1:

    def Sum(self, lst):
        sum1 = 0
        for x in lst:
            n = int(x)
            sum1 += n
        print("sum of", lst, "=", sum1)


lst = [1,2, 3, 4, 5]
obj = List1()
obj.Sum(lst)
