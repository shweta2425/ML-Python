# Write a python program to check whether two lists are circularly identical.

lst1 = [10, 1, 1, 10]
lst2 = [1, 1, 10, 10]
print(' '.join((map(str, lst2))) in ' '.join(map(str, lst1 * 2)))

