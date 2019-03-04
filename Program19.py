# Write a Python program to get file creation and modification date/times

import os.path, time
#print last modification time
print("Last modified: %s" % time.ctime(os.path.getmtime("Program1.py")))
#print creation time
print("Created: %s" % time.ctime(os.path.getctime("Program1.txt")))
