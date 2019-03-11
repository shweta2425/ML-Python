# Write a Python program to get the smallest number from a list
class List3:

    def findsmall(self, lst):
        small = lst[0]
        for i in lst:
            if small > i:
                small = i
        print("smallest element is ", small)


lst = [10, 37, 483, 88]
print("smallest", min(lst))
obj = List3()
obj.findsmall(lst)
