# 4. Write a program to multiply matrices in problem 1

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


class MulMatrices(Parent):
    def __int__(self):
        pass

    # creating Utility class object
    obj = User()

    def CreateMatrix(self):

        # calling multiplication of matrix method by passing parent class variables(matrices) using Utility class object
        mulMatrix = self.obj.MulMatrix(self.A, self.B)

        # calling display method by using class object
        obj1.display(mulMatrix)

    def display(self, mulMatrix):

        # printing matrix A
        print("Matrix A :")
        for item in self.A:
            print(item)

        # printing matrix B
        print("Matrix B :")
        for item in self.B:
            print(item)

        # printing Multiplication Matrix
        print("Multiplication of Matrix A & B")
        for item in mulMatrix:
            print(item)


# creating object of class
obj1 = MulMatrices()

# calling method by using class object
obj1.CreateMatrix()
