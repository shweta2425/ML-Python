# Write a Python program to remove the first occurrence of a specified element from an array.
from Week2.Utilities import utility


class Array4:
    # arr = array.array('i', [10, 20, 30, 40, 50])
    # creating class obj
    obj = utility.User()
    # Accepts array from user
    arr1 = obj.accepts()

    def Remove(self):
        try:
            num = input("enter ele to be deleted")
            # remove specified ele
            temp = utility.User.CheckInt(num)
            if temp:
                self.arr1.remove(num)
                print("\n after removing val :")
                for i in range(len(self.arr1)):
                    print(self.arr1[i], end=" ")
            else:
                print("Enter only integers")
        except Exception as e:
            print(e)


obj = Array4()
obj.Remove()
