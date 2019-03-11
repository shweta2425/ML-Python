# Write a Python program to multiplies all the items in a list.
class List2:

    def Multiplication(self, lst):
        sum1 = 1
        for x in lst:
            n = int(x)
            sum1 *= n
        print("Multiplication of", lst, "=", sum1)


lst = [1, 2, 3, 4, 5]
obj = List2()
obj.Multiplication(lst)
