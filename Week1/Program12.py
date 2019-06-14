# Write a python program to call an external command in Python.

from subprocess import call
#list the contents of current directory
call(["ls", "-l"])