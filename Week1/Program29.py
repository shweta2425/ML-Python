# Write a Python program to get the name of the host on which the routine is running

import socket
#return hostname where py interpreter is currently executing
print(socket.gethostname())