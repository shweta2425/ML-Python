# Write a Python program to clone or copy a list.
class List7:
    

    def Copy(self, lst):
        cpy = lst.copy()
        print("Copy of list:", cpy)


lst = [1, 2, 3, 53, 332]
obj = List7()
obj.Copy(lst)
