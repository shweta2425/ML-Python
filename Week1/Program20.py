# Write a Python program to sort three integers without using conditional statements and loops

x = int(input("Input first number: "))
y = int(input("Input second number: "))
z = int(input("Input third number: "))
#return min num
a1 = min(x, y, z)
#return max num
a3 = max(x, y, z)
#return remaining num out of 3 numbers
a2 = (x + y + z) - a1 - a3
print("Numbers in sorted order: ", a1, a2, a3)