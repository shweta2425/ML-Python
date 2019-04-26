# Write a Python program to sort files by date

import glob
import os
#return list of file
files = glob.glob("*.py")
#sort file according to modification time of files
files.sort(key=os.path.getmtime)
print("\n".join(files))