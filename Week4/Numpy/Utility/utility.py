import numpy as np


class User:

    def convert_list_to_array(self, list1):
        #  creates array using given sequence of list
        arr = np.asarray(list1, dtype=float)
        return arr

    def arrange(self):
        # creates array using given iterator
        arr = np.arange(2, 11)
        b = arr.reshape(3, 3)
        return b

    def create_null_array(self):
        # creates null array of specified size
        arr = np.zeros((1, 10), dtype=float)
        return arr

    def reverse(self, arr):
        # reverse the array using slicing
        arr = arr[::-1]
        return arr

    def create_ones(self):
        # return identity matrix of specified size
        arr = np.ones((5, 5), dtype=int)
        return arr

    def create_array_from_LandT(self, list1, tuple1):
        # creates array from list
        arr = np.asarray(list1, dtype=int)
        # creates array from tuple
        arr1 = np.asarray(tuple1, dtype=int)
        # modify array with specified size
        b = arr1.reshape((2, 3))
        return arr, b

    def intersection(self, arr1, arr2):
        # return intersection of two arrays
        arr = np.intersect1d(arr1, arr2)
        return arr

    def find_unique_set(self, arr1, arr2):
        # returns the rows from arr1 that are not in arr2
        arr = np.setdiff1d(arr1, arr2)
        return arr

    def find_set_exclusive(self, arr1, arr2):
        # Find the set exclusive-or of two arrays
        arr = np.setxor1d(arr1, arr2)
        return arr

    def compare_array(self, arr1, arr2):
        # return true if arr1 is greater than arr2
        big = np.greater(arr1, arr2)
        # return true if arr1 is less than arr2
        small = np.less(arr1, arr2)
        # return true if arr1 is greater than equal to arr2
        big_equal = np.greater_equal(arr1, arr2)
        # return true if arr1 is less than equal to arr2
        small_equal = np.less_equal(arr1, arr2)
        return big, small, big_equal, small_equal

    def create_flatten_array(self, arr):
        # Return a copy of the array collapsed into one dimension.
        arr1 = arr.flatten()
        return arr1

    def conactenate_array(self, arr1, arr2):
        # concatenate arr1 with arr2
        concat_array = np.concatenate((arr1, arr2), 1)
        return concat_array

    def convert_to_list(self, arr):
        # convert array into list type
        list1 = arr.tolist()
        return list1