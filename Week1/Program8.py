# Write a Python program to create a histogram from a given list of integers

l=[2,3,4,5]

for n in l:
    output = ''
    times=n
    while(times>0):
        output=output+'*'
        times=times-1
    print(output)
