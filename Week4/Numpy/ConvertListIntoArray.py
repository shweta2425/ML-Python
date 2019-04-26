import numpy as np
from Week4.Numpy.Utility.utility import User


class convert_list:
    def __init__(self):
        # initializing
        self.list1 = [12.23, 13.32, 100, 36.32]
        self.list2 = [1, 2, 3, 4, 5, 6, 7, 8]
        self.list3 = [40, 50, 60, 70, 80, 90]
        self.tuple1 = tuple((1, 2, 3, 8, 4, 6,))

    def __del__(self):
        print("object destroyed")

    # creates utility class object
    utility_obj = User()

    def create_array(self):

        # calling method to create array
        arr = self.utility_obj.convert_list_to_array(self.list1)
        # calling method to display array
        obj.display(arr)

    def arrange(self):

        # calling method to create array
        arr = self.utility_obj.arrange()
        # calling method to display array
        obj.display(arr)

    def null_array(self):

        # calling method to create array
        arr = self.utility_obj.create_null_array()
        # calling method to display array
        obj.display(arr)
        # updates value
        arr[0, 6] = 11
        obj.display(arr)

    def reverse_array(self):
        # creates array
        arr = np.arange(12, 38)
        # calling method to display array
        obj.display(arr)
        arr = self.utility_obj.reverse(arr)
        # calling method to display reverse array
        print("reverse array is")
        obj.display(arr)

    def border_one(self):
        arr = self.utility_obj.create_ones()
        # calling method to display array
        obj.display(arr)
        arr[1:-1, 1:-1] = 0
        # calling method to display array
        obj.display(arr)

    def fill_border_zero(self):
        arr = np.ones((3, 3))
        # calling method to display array
        obj.display(arr)
        # it gives padding(element) around the matrix
        x = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
        # calling method to display array
        obj.display(x)

    def checkborder(self):

        arr = np.zeros((8, 8), dtype=int)
        # slicing to get check board
        arr[1::2, ::2] = 1
        arr[::2, 1::2] = 1

        # calling method to display array
        obj.display(arr)

    def create_arrays(self):
        # calling method to create array
        arr = self.utility_obj.create_array_from_LandT(self.list2, self.tuple1)
        # calling method to display array
        obj.display(arr)

    def append_array(self):
        # creates an array
        arr = np.array([10, 20, 30])
        # calling method to display array
        obj.display(arr)
        # append values(list) to the end of array
        newArr = np.append(arr, self.list3)
        # calling method to display array
        obj.display(newArr)

    def find_real_imag(self):

        # creates an array
        arr = np.array([1.00000000 + 0.j, 0.70710678 + 0.70710678j], dtype=complex)
        # calling method to display array
        obj.display(arr)
        # getting real part of num
        real = arr.real
        # return imaginary part of num
        imaginary = arr.imag

        print("real =", real)
        print("imaginary=", imaginary)

    def array_info(self):
        # calling method to create array
        arr = self.utility_obj.convert_list_to_array(self.list1)
        # calling method to display array
        obj.display(arr)
        print("array size =", arr.size)
        print("each array element of", arr.itemsize, "bytes")
        print("total bytes consumed by the elements", arr.nbytes)

    def find_common(self):

        # creates an array
        arr1 = np.array([0, 10, 20, 40, 60])
        arr2 = np.array([10, 30, 40])
        # calling method to find common elements btw two arrays
        arr = self.utility_obj.intersection(arr1, arr2)
        print("common elements are")
        # calling method to display array
        obj.display(arr)

    def find_unique_set(self):

        # creates an array
        arr1 = np.array([0, 10, 20, 40, 60, 80])
        arr2 = np.array([10, 30, 40, 50, 70, 90])
        arr = self.utility_obj.find_unique_set(arr1, arr2)
        print("array 1")
        # calling method to print 1st array
        obj.display(arr1)
        print("array 2")
        # calling method to print 2st array
        obj.display(arr2)
        print("set difference between two arrays")
        # calling method to display array
        obj.display(arr)

    def set_exclusive(self):

        # creates an array
        arr1 = np.array([0, 10, 20, 40, 60, 80])
        arr2 = np.array([10, 30, 40, 50, 70])
        # calling method to find set exclusive of arrays
        arr = self.utility_obj.find_set_exclusive(arr1, arr2)
        print("array 1")
        # calling method to print 1st array
        obj.display(arr1)
        print("array 2")
        # calling method to print 2st array
        obj.display(arr2)
        print("the set exclusive-or of two arrays")
        # calling method to display array
        obj.display(arr)

    def display(self, arr):
        print("Array =\n", arr)

    def menu(self):
        print("1.convert list into array ")
        print("2. create matrix with values ranging from 2 to 10")
        print("3.create null array")
        print("4.reverse array")
        print("5.create array with border 1")
        print("6.create array with border 0")
        print("7.check board")
        print("8.create array from list & tuple")
        print("9.append values to the end of the array")
        print("10.find the real and imaginary parts of an array of complex numbers")
        print("11.print array size")
        print("12.find intersection of two arrays")
        print("13.find the set difference of two arrays")
        print("14.find the set exclusive-or of two arrays")
        print("0.exit")

        flag = False

        while not flag:
            try:
                choice = int(input("\nEnter ur choice"))
                if not choice <= 0 and choice <= 14:

                    if choice == 1:
                        obj.create_array()

                    if choice == 2:
                        obj.arrange()

                    if choice == 3:
                        obj.null_array()

                    if choice == 4:
                        obj.reverse_array()

                    if choice == 5:
                        obj.border_one()

                    if choice == 6:
                        obj.fill_border_zero()

                    if choice == 7:
                        obj.checkborder()

                    if choice == 8:
                        obj.create_arrays()

                    if choice == 9:
                        obj.append_array()

                    if choice == 10:
                        obj.find_real_imag()

                    if choice == 11:
                        obj.array_info()

                    if choice == 12:
                        obj.find_common()

                    if choice == 13:
                        obj.find_unique_set()

                    if choice == 14:
                        obj.set_exclusive()

                    if choice == 0:
                        flag = True
                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input and Try again")


# creates class object
obj = convert_list()
flag = False
# calling method by using class object
obj.menu()
