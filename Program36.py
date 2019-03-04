# Write a Python program to determine if variable is defined or not.

x = 'aa'
#checks if it contains any error
try:
  x > 1
#execute if try block raises name error
except NameError:
  print("You have a variable that is not defined.")
#execute if try block raises data type error
except TypeError:
  print("You are comparing values of different type")
else:
  print("The 'Try' code was executed without raising any errors!")
