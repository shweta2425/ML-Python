class User:
    def __int__(self):
        print(" ")

    def AddMatrix(self, A, B):
        # returns addition of two matrices
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def MulScalarMatrix(self, X, Y):
        # Multiplies each matrix element by number
        return [[X[i][j] * Y for j in range(len(X[0]))] for i in range(len(X))]

    def MulVectorMatrix(self, X, Y):
        #
        result = [0] * len(X)

        sum1 = 0
        for i in range(len(X)):
            r = X[i]
            # iterate through columns of Y
            for j in range(len(Y)):
                sum1 += sum(r[j] * Y[j])
            result[i] = sum1
            sum1 = 0
        return result

    def MulMatrix(self, A, B):
        # returns addition of two matrices
        return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def transpose(self, Y):
        # returns transpose of matrix
        transposeMatrix = [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]
        # iterate through row of Y
        for i in range(len(Y)):
            # iterate through column of Y
            for j in range(len(Y[0])):
                # replacing row into column vice versa
                transposeMatrix[i][j] = Y[j][i]

        return transposeMatrix

    def determinant(self, matrix):
        # return determinant of matrix
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        return (matrix[0][0] * ((matrix[1][1] * matrix[2][2]) - (matrix[2][1] * matrix[1][2]))
                - matrix[0][1] * ((matrix[1][0] * matrix[2][2]) - (matrix[2][0] * matrix[1][2]))
                + matrix[0][2] * ((matrix[1][0] * matrix[2][1]) - (matrix[2][0] * matrix[1][1])))

    # def get_matrix_minor(self, m, i, j):
    #     return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]
    #
    # def get_inverse_matrix(self, m):
    #     determinant = self.determinant(m)
    #     # special case for 2x2 matrix:
    #     if len(m) == 2:
    #         return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
    #                 [-1 * m[1][0] / determinant, m[0][0] / determinant]]
    #
    #     co_factors = []
    #     for r in range(len(m)):
    #         co_factor_row = []
    #         for c in range(len(m)):
    #             minor = self.get_matrix_minor(m, r, c)
    #             co_factor_row.append(((-1) ** (r + c)) * self.determinant(minor))
    #         co_factors.append(co_factor_row)
    #     co_factors = self.transpose(co_factors)
    #     for r in range(len(co_factors)):
    #         for c in range(len(co_factors)):
    #             co_factors[r][c] = co_factors[r][c] / determinant
    #     return co_factors
    # -------------
    # def transpose_matrix(self, matrix_1):
    #     result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    #     for row in range(len(matrix_1)):
    #         for col in range(len(matrix_1[0])):
    #             result[col][row] = matrix_1[row][col]
    #     return result

    @staticmethod
    def get_matrix_minor(m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    # @staticmethod
    # def get_matrix_determinant(m):
    #     # base case for 2x2 matrix
    #     if len(m) == 2:
    #         return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    #
    #     determinants = (m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
    #                     - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
    #                     + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]))
    #     return determinants

    def get_inverse_matrix(self, m):
        determinant = self.determinant(m)
        # special case for 2x2 matrix:
        if len(m) == 2:
            return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                    [-1 * m[1][0] / determinant, m[0][0] / determinant]]

        # find matrix of co_factors
        co_factors = []
        for r in range(len(m)):
            co_factor_row = []
            for c in range(len(m)):
                minor = self.get_matrix_minor(m, r, c)
                co_factor_row.append(((-1) ** (r + c)) * self.determinant(minor))
            co_factors.append(co_factor_row)
        co_factors = self.transpose(co_factors)
        for r in range(len(co_factors)):
            for c in range(len(co_factors)):
                co_factors[r][c] = co_factors[r][c] / determinant
        return co_factors


