# Write a Python program to print the calendar of a given month and year.
# Note : Use 'calendar' module.

import calendar
y = int(input("Input the year : "))
m = int(input("Input the month : "))
# print the calendar of a given month and year
print(calendar.month(y, m))