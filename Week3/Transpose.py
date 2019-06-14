# 6. Write a program to find transpose matrix of matrix Y in problem 1

from Week3.utility.utility import User


class Parent:

    # matrix Y
    Y = [[5, 8, 1],
         [6, 7, 3],
         [4, 5, 9]]


class Transpose(Parent):
    def __int__(self):
        pass

    # creating Utility class object
    obj = User()

    def CreateMatrix(self):

        # calling transpose of matrix method by passing parent class variables(matrices) using Utility class object
        transposeMatrix = self.obj.transpose(self.Y)

        # calling display method by using class object
        obj1.display(transposeMatrix)

    def display(self, transposeMatrix):

        # printing matrix Y
        print(" Matrix Y")
        for item in self.Y:
            print(item)

        # printing transpose of matrix Y
        print("\nTranspose of Y")
        for r in transposeMatrix:
            print(r)


# creating object of class
obj1 = Transpose()

# calling method by using class object
obj1.CreateMatrix()
