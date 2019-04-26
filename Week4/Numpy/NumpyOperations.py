import numpy as np
from Week4.Numpy.Utility.utility import User


class NumpyOperations:
    def __init__(self):
        self.list1 = [[1, 2, 3, 4, 10, 20, 100], [11, 22, 33]]

    def __del__(self):
        print("class object destroyed")

    # creates utility class object
    utility_obj = User()

    def compare(self):
        # creates an array
        arr1 = np.asarray([1, 2])
        arr2 = np.asarray([4, 5])
        # calling function to compare arrays
        arr = self.utility_obj.compare_array(arr1, arr2)

        # printing arr1
        print("a =", arr1)

        # printing arr2
        print("b =", arr2)

        # printing returned array
        print("a > b ", arr[0][0], arr[0][1])
        print("a < b ", arr[1][0], arr[1][1])
        print("a >= b ", arr[2][0], arr[2][1])
        print("a <= b ", arr[3][0], arr[3][1])

    def savetxt(self):
        np.savetxt('abc.txt', self.arr, delimiter=" ")
        # open file in read mode
        a = open("abc.txt", 'r')
        print("the file contains:")
        # reads file contains and print it on terminal
        print(a.read())

    def flatten_array(self):
        # # calling method to create array
        # arr = self.utility_obj.convert_list_to_array(self.list1)
        # print(arr)
        # # calling method to create flatten array
        # new_arr = self.utility_obj.create_flatten_array(arr)
        # # calling method to display array
        # obj.display(new_arr)

        arr = np.array([[10, 20, 30], [20, 40, 50]])
        # calling method to display array
        obj.display(arr)

        # creates flatten array
        flatten_array = arr.flatten()
        print("flatten array")
        # calling method to display array
        obj.display(flatten_array)

    def change_dt(self):
        arr = np.array([[2, 4, 6], [6, 8, 10]], np.int32)
        print("data type of array arr  is", arr.dtype)
        arr1 = arr.astype(np.float64)
        print("data type of array arr  is", arr1.dtype)

    def create_identity(self):
        # creates identity matrix of given size
        arr = np.eye(3)
        # calling method to display array
        obj.display(arr)

    def concat(self):
        arr1 = np.array([[0, 1, 3], [5, 7, 9]], dtype=int)
        arr2 = np.array([[0, 2, 4], [6, 8, 10]], dtype=int)
        concat_array = self.utility_obj.conactenate_array(arr1, arr2)
        # calling method to display array
        obj.display(concat_array)

    def immutable_array(self):
        # creates null matrix of given size
        arr = np.zeros((3, 3))
        # calling method to display array
        obj.display(arr)
        # makes an array read only
        arr.flags.writeable = False
        print("we will try to modify value of immutable array")
        arr[0][0] = 1
        # calling method to display array
        obj.display(arr)

    def mul_array(self):
        arr = np.arange(0, 12)
        a = arr.reshape((3, 4))
        # calling method to display array
        obj.display(a)
        # iterate over an array

        for item in np.nditer(a, op_flags=['readwrite']):
            item[...] = 3 * item

        print("New array is:")

        # calling method to display array
        obj.display(a)

    def convert_array_list(self):
        arr = np.array([[0, 1], [2, 3], [4, 5]])
        # calling method to display array
        obj.display(arr)
        list1 = self.utility_obj.convert_to_list(arr)
        print("list is", list1)

    def convert_list_with_precision(self):
        arr = np.array([0.26153123, 0.52760141, 0.5718299, 0.5927067, 0.7831874, 0.69746349, 0.35399976, 0.99469633,
                        0.0694458, 0.54711478])
        # calling method to display array
        obj.display(arr)
        list1 = self.utility_obj.convert_to_list(arr)
        arr1 = self.utility_obj.convert_list_to_array(list1)
        np.set_printoptions(precision=3)
        # calling method to display array
        obj.display(arr)

    def supress(self):
        arr = np.array([1.60000000e-10, 1.60000000e+00, 1.20000000e+03, 2.35000000e-01])
        # calling method to display array
        obj.display(arr)
        np.set_printoptions(suppress=True)
        # calling method to display array
        obj.display(arr)

    def add_column(self):
        arr1 = np.array([[10, 20, 30], [40, 50, 60]])
        arr2 = np.array([[100], [200]])
        new_arr = np.append(arr1, arr2, axis=1)
        # calling method to display array
        obj.display(new_arr)

    def remove_element(self):
        arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # calling method to display array
        print("array before removing elements")
        obj.display(arr)
        index = [0, 3, 4]
        new_arr = np.delete(arr, index)
        obj.display(new_arr)

    def display(self, arr):
        print("Array =\n", arr)

    def menu(self):
        print("1.compare two arrays using numpy")
        print("2.save a NumPy array to a text file")
        print("3.create flatten array")
        print("4.change data type")
        print("5.create a 3-D array with ones on a diagonal")
        print("6.19.incomplete")
        print("7.concatenate arrays")
        print("8.create immutable array")
        print("9.multiply whole array by 3")
        print("10.convert a NumPy array into Python list structure")
        print("11.convert a NumPy array into Python list structure & print values with precision 2")
        print("12.suppresses the use of scientific notation for small numbers")
        print("13.add an extra column to an numpy array")
        print("14.remove specific elements in a numpy array")

        print("0.exit")
        flag = False

        while not flag:
            try:
                choice = int(input("\nEnter ur choice"))
                if not choice <= 0 and choice <= 15:

                    if choice == 1:
                        obj.compare()
                    if choice == 2:
                        obj.savetxt()
                    if choice == 3:
                        obj.flatten_array()
                    if choice == 4:
                        obj.change_dt()
                    if choice == 5:
                        obj.create_identity()
                    if choice == 6:
                        pass
                    if choice == 7:
                        obj.concat()
                    if choice == 8:
                        obj.immutable_array()

                    if choice == 9:
                        obj.mul_array()

                    if choice == 10:
                        obj.convert_array_list()

                    if choice == 11:
                        obj.convert_list_with_precision()

                    if choice == 12:
                        obj.supress()

                    if choice == 13:
                        obj.add_column()

                    if choice == 14:
                        obj.remove_element()

                    if choice == 0:
                        flag = True
                else:
                    raise ValueError
            except ValueError:
                print("\nPlease give valid input and Try again")


# creates class object
obj = NumpyOperations()
flag = False
# calling method by using class object
obj.menu()
