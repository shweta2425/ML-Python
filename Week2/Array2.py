# Write a Python program to reverse the order of the items in the array.


from Week2.Utilities import utility


class Array2:
    flag = True
    while flag:
        print("\n1.to get the last part of a string before a specified character of str")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice == 0 or choice == 1:
                if choice == 1:

                    # creating class obj
                    obj = utility.User()
                    # Accepts array from user
                    arr1 = obj.accepts()

                    def Reverese(self):
                        print("Array before reversing is :", end="")
                        for i in range(len(self.arr1)):
                            print(self.arr1[i], end=" ")
                        #     reverse the array
                        self.arr1.reverse()
                        print("\nThe array after reversing is : ", end="")
                        for i in range(len(self.arr1)):
                            print(self.arr1[i], end=" ")

                if choice == 0:
                    print("\nThank You !!!!!!!")
                    flag = False

            else:
                raise ValueError
        except ValueError:
            print("\nPlease give valid input")
            print("Try again.....")

obj = Array2()
obj.Reverese()
