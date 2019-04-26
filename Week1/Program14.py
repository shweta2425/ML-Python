# Write a Python program to list all files in a directory in Python.

from os import listdir
from os.path import isfile, join
#return list of entries in dir
files_list = [f for f in listdir('/home/admin1/PycharmProjects/Basic Python/')
#if its file will add in files_list
              if isfile(join('/home/admin1/PycharmProjects/Basic Python/', f))]
#print file names
print(files_list);