# Write a Python program which accepts a sequence of comma-separated numbers from user
# and generate a list and a tuple with those numbers.
# Sample data : 3, 5, 7, 23
# Output :
# List : ['3', ' 5', ' 7', ' 23']
# Tuple : ('3', ' 5', ' 7', ' 23')


values = input("Enter Numbers with comma")

# split string into comma separated list
list = values.split(",")
tuple = tuple(list)

print('List : ',list)
print('Tuple : ',tuple)