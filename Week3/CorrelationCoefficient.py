# The table below shows the height, x, in inches and the pulse rate, y, per minute,
# for 9 people. Write a program to find the correlation coefficient and interpret your result.
# 	x ⇒ 68 72 65 70 62 75 78 64 68
# 	y ⇒ 90 85 88 100 105 98 70 65 72


class CorrelationCoefficicent:
    def __init__(self):
        self.listX = [68, 72, 65, 70, 62, 75, 78, 64, 68]
        self.listY = [90, 85, 88, 100, 105, 98, 70, 65, 72]

    def findSum(self):
        lenghthX = len(self.listX)
        lenghthY = len(self.listY)
        sumX = 0
        sumY = 0
        for item in self.listX:
            sumX = sumX + item
        print(sumX)

        for item in self.listY:
            sumY = sumY + item

        sumationX = sumX / lenghthX
        sumationY = sumY / lenghthY
        print(sumationX, sumationY)
        obj.findxy(sumationX, sumationY)

    def findxy(self, sumationX, sumationY):
        listx=[]
        listy=[]
        sumx = 0
        sumy = 0

        for item in self.listX:
            listx=item-sumationX

        for item in self.listY:
            listy = item - sumationY

        for item in listx:
            sumx=sumx+item

        for item in listy:
            sumy=sumy+item




obj = CorrelationCoefficicent()
obj.findSum()
