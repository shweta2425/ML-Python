# 5. Write a program to find inverse matrix of matrix X in problem 1
from Week3.utility.utility import User


class Parent:
    # X = [[12, 7, 3],
    #      [4, 5, 6],
    #      [7, 8, 9]]
    X = [[1, 2, 3],
         [0, 1, 4],
         [5, 6, 0]]


class Inverse(Parent):
    def __int__(self):
        pass

    obj = User()

    def CreateMatrix(self):

        deteminant = self.obj.determinant(self.X)

        # obj1.display(transposeMatrix)

        if deteminant == 0:
            print("Matrix X doesn't have Inverse matrix")

        else:
            print("Matrix X =")
            for item in self.X:
                print(item)
            print("Determinant of matrix X =", deteminant)
            transposeMatrix = self.obj.transpose(self.X)
            print("Transpose of matrix X :")
            for item in transposeMatrix:
                print(item)
            print("Inverse matrix")
            inverseMatrix = self.obj.get_inverse_matrix(transposeMatrix)
            for item in range(len(inverseMatrix)):
                print(inverseMatrix[item])


obj1 = Inverse()
obj1.CreateMatrix()
