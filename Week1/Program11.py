# Write a Python program to check whether a file exists.

import os.path
#opens file in write mode
open('shweta.txt', 'w')
#return True if file exist o.w false
print(os.path.isfile('shweta.txt'))

