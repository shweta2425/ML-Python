# Write a Python program to get numbers divisible by fifteen from a list using an anonymous function

lst = []
n = int(input('How many numbers: '))
for i in range(n):
    num = int(input('Enter number '))
    lst.append(num)

result=list(filter(lambda x:(x%15==0),lst))
print("Number divisible by 15 are",result)