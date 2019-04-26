# Write a program to get execution time for a Python method


import time
#return current system time in ticks since the epoch
start_time = time.time()
#print required time to execute program
print("--- %s seconds ---" % (time.time() - start_time))
