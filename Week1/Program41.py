# Write a Python program to convert an integer to binary keep leading zeros.
# Sample data : 50
# Expected output : 00001100, 0000001100

n=int(input("enter a num"))

print(format(n, '08b'))  # 8 bit binary representation of str
print(format(n, '010b')) #10 bit binary representation of str

