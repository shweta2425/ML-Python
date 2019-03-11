from array import *


class Array1:

    def __int__(self):
        print("this is default constructor")

    def disp(self):
        flag = True
        arr1 = array('i', [])
        while flag:
            n = int(input("enter no of elements"))
            for i in range(n):
                ele = int(input("enter element"))
                arr1.append(ele)

            for i in range(n):
                print("element at", i, "th index", arr1[i])

            else:
                print("\n1.to continue")
                print("0.to exit")
                ch = int(input())

            if ch == 0:
                flag = False


obj = Array1()
obj.disp()

# arr = array('i', [10, 20, 30, 40, 50])
# print(arr[0])
# print(arr[1])
# print(arr[2])
# print(arr[3])
# print(arr[4])
# print(arr)


# arr1 = array('i', [])
#
# NoOfEle = int(input("enter no of elements"))
#
# for i in range(NoOfEle):
#     ele = int(input("enter element"))
#     arr1.append(ele)
# 
# for i in range(NoOfEle):
#     print(arr1[i])
