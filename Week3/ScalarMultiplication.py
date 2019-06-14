# 2. Write a program to perform scalar multiplication of matrix and a number
#  	X = [[12,7,3],
#     	[4 ,5,6],
#     	[7 ,8,9]]
#  Y = 9
from Week3.utility.utility import User


class Parent:
    # matrix X

    X = [[12, 7, 3],
         [4, 5, 6],
         [7, 8, 9]]
    # storing a number into the var
    Y = 9


class ScalarMultiplication(Parent):
    def __int__(self):
        pass

    # creating Utility class object
    obj = User()

    def ScalarMatrix(self):

        # calling addition of matrix method by passing parent class variables(matrices) using Utility class object
        Scalarmultiplication = self.obj.MulScalarMatrix(self.X, self.Y)

        # calling display method by using class object
        obj1.display(Scalarmultiplication)

    def display(self, Scalarmultiplication):

        # printing matrix X
        print("Matrix X :")
        for item in self.X:
            print(item)

        # print number
        print("Number =", self.Y)

        # printing scalar multiplication matrix
        print("\nScalar Multiplication :")
        for item in Scalarmultiplication:
            print(item)


# creating object of class
obj1 = ScalarMultiplication()

# calling method by using class object
obj1.ScalarMatrix()

