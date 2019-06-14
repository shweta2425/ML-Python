# Write a Python function to find the maximum and minimum numbers from a sequence of numbers.
# Note: Do not use built-in functions
lst = []
n = int(input('How many numbers: '))
for i in range(n):
    num = int(input('Enter number '))
    lst.append(num)
#assign lst[0] as large and small num
    l = lst[0]
    s = lst[0]

    for num in lst:
#checks if num is greater than l if yes then assigning it as a large
        if num > l:
            l = num
#checks if num is smaller than s if yes then assigning it as a smallest
        elif num < s:
            s = num
print("small=",s,"large=",l)
