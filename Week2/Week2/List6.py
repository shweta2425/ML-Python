# Write a Python program to remove duplicates from a list

class List6:
    def Duplicate(self, lst):
        mylst = list(dict.fromkeys(lst))
        print(mylst)


lst = [1, 2, 3, 4, 1, 1]
obj=List6()
obj.Duplicate(lst)
