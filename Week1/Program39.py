# Write a Python program to find files and skip directories of a given directory

import os
#return list of files
print([f for f in os.listdir('/home/admin1/PycharmProjects/Basic Python')
#checks if file and joins it
       if os.path.isfile(os.path.join('/home/admin1/PycharmProjects/Basic Python', f))])

