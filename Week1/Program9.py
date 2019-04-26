# Write a Python program to concatenate all elements in a list into a string and return it.

def concatenate_list_data(list):
    result= ''
    for element in list:
        # convert int element into string
        result += str(element)
    return result

print(concatenate_list_data([1, 5, 12, 2]))

