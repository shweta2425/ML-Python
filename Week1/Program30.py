# Write a Python program to access and print a URL's content to the console

import urllib.request

link = "https://www.google.com"

f = urllib.request.urlopen(link)
myfile = f.read()
print(myfile)
