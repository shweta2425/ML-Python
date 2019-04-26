# Write a Python program to get the command-line arguments
#(name of the script, the number of arguments, arguments) passed to a script

import sys
#return name of py script
print("This is the name/path of the script:",sys.argv[0])
#return lenght of the argv array
print("Number of arguments:",len(sys.argv))
#return list items
print("Argument List:",str(sys.argv))
