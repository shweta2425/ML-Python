# Write a Python program to print a specified list after removing the 0th, 4th and 5th elements.
# Sample List : ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']
# Expected Output : ['Green', 'White', 'Black']



class List10:

    def Delete(self, lst):
        del lst[0]
        del lst[4]
        del lst[3]

        print(lst)


lst = ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']
obj = List10()
obj.Delete(lst)
