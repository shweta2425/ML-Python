# 3. Write a program to perform multiplication of given matrix and vector
# 			X = [[ 5, 1 ,3], [ 1, 1 ,1], [ 1, 2 ,1]]
# 			Y = [1, 2, 3]
from Week3.utility.utility import User

class VectorMultiplication:
    # matrix X
    X = [[5, 1, 3], [1, 1, 1], [1, 2, 1]]

    # matrix Y
    Y = [[1], [2], [3]]

    def __int__(self):
        pass

    # creating Utility class object
    obj = User()

    def CreateMatrix(self):

        # calling vector matrix multiplication method by passing parent class variables using Utility class object
        vectorMatrix = self.obj.MulVectorMatrix(self.X, self.Y)

        # calling display method by using class object
        obj1.display(vectorMatrix)

    def display(self, vectorMatrix):

        # printing matrix X
        print(" Matrix X")
        for item in self.X:
            print(item)

        # printing matrix Y
        print("\nVector Y")
        for item in self.Y:
            print(item)

        # printing vector matrix multiplication of matrix
        print("\nVector matrix multiplication is :")
        print(vectorMatrix)


# creating object of class
obj1 = VectorMultiplication()

# calling method by using class object
obj1.CreateMatrix()
