# 1. Write a python program to add below matrices
#    			X = [[12,7,3],
#     			     [4 ,5,6],
#     			     [7 ,8,9]]
# 			    Y = [[5,8,1],
#     			     [6,7,3],
#     			     [4,5,9]]

from Week3.utility.utility import User


class Parent:
    # matrix A

    A = [[12, 7, 3],
         [4, 5, 6],
         [7, 8, 9]]

    # matrix B

    B = [[5, 8, 1],
         [6, 7, 3],
         [4, 5, 9]]

    # A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # B = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


class AddMatrices(Parent):
    def __int__(self):
        pass

    # creating Utility class object
    obj = User()

    def CreateMatrix(self):

        # calling addition of matrix method by passing parent class variables(matrices) using Utility class object
        Addition = self.obj.AddMatrix(self.A, self.B)

        # calling display method by using class object
        obj1.display(Addition)

    def display(self, Addition):

        # printing matrix A
        print("Matrix A :")
        for item in self.A:
            print(item)

        # printing matrix B
        print("Matrix B :")
        for item in self.B:
            print(item)

        # printing addition matrix
        print("Addition of Matrix A & B")
        for item in Addition:
            print(item)


# creating object of class
obj1 = AddMatrices()

# calling method by using class object
obj1.CreateMatrix()
