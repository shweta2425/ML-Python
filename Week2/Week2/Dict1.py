# 1.Write a Python script to sort (ascending and descending) a dictionary by value.
# 2.Write a Python script to add a key to a dictionary.
#
# 	Sample Dictionary : {0: 10, 1: 20}
# 	Expected Result : {0: 10, 1: 20, 2: 30}
# 3. Write a Python script to concatenate following dictionaries to create a new one.
#
# 	Sample Dictionary :
# 	dic1={1:10, 2:20}
# 	dic2={3:30, 4:40}
# 	dic3={5:50,6:60}
# 	Expected Result : {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}
# 4. Write a Python program to iterate over dictionaries using for loops.
# 5. Write a Python script to generate and print a dictionary that contains
# a number (between 1 and n) in the form (x, x*x).
# Sample Dictionary ( n = 5) :
# Expected Output : {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
# 6. Write a Python program to remove a key from a dictionary.
# 7. Write a Python program to print all unique values in a dictionary.
# Data :[{"V":"S001"}, {"V": "S002"}, {"VI": "S001"}, {"VI": "S005"}, {"VII":"S005"}, {"V":"S009"},{"VIII":"S007"}]
# Expected Output : Unique Values: {'S005', 'S002', 'S007', 'S001', 'S009'}
# 8.Write a Python program to create a dictionary from a string.
# 	Note: Track the count of the letters from the string.
# 	Sample string : 'w3resource'
# 	Expected output: {'3': 1, 's': 1, 'r': 2, 'u': 1, 'w': 1, 'c': 1, 'e': 2, 'o': 1}
# 9. Write a Python program to print a dictionary in table format.
# 10. Write a Python program to count the values associated with key in a dictionary.
# 	Sample data: = [{'id': 1, 'success': True, 'name': 'Lary'},
# 	{'id': 2, 'success': False, 'name': 'Rabi'}, {'id': 3, 'success': True, 'name': 'Alex'}]
# 	Expected result: Count of how many dictionaries have success as True

# 11. Write a Python program to convert a list into a nested dictionary of keys.
# 12. Write a Python program to check multiple keys exists in a dictionary.
# 13.Write a Python program to count number of items in a dictionary value that is a list.

from Week2.Utilities import utility


class Dict1:
    flag = True
    while flag:
        print("\n 1.sort dictionary in asc & desc order")
        print("2.Add new key in dict")
        print("3.Concatenate dictionaries")
        print("4.Iterate over dict")
        print("5.print dict in the form (x, x*x)")
        print("6.Remove key")
        print("7.Print unique values")
        print("8.create dictionary from string")
        print("9.Print dict in table format")
        print("10.Count val associated with key")
        print("11.Convert list into nested dict")
        print("12.Check if multiple keys exists in a dictionary")
        print("13.Check no.of lists & items in dict")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice >= 0 and choice <= 13:
                dict1 = {0: 10, 1: 20, 2: 1993}
                print(dict1)

                if choice == 0:
                    flag = False

                if choice == 1:
                    asc, desc = utility.User.SortDict(dict1)

                    print("Ascending", asc)
                    print("Descending", desc)

                # for x in d.items():
                #     print(x)

                if choice == 2:
                    # adds key in the dictionary
                    dict1[4] = 25
                    print("After adding new key", dict1)

                if choice == 3:
                    dict2 = {3: 30, 4: 40}
                    dict3 = {5: 50, 6: 60}
                    # creates dictionary
                    newDict = dict()
                    # add dictionary in newDict(dictionary)
                    for dictionary in (dict1, dict2, dict3):
                        newDict.update(dictionary)
                    print("after concatenation", newDict)

                if choice == 4:
                    # prints keys & values of dictionary
                    for key, val in dict1.items():
                        print("key:", key, "value:", val)

                if choice == 5:
                    n = int(input("how many keys u wanna generate"))
                    # creates dictionary
                    newDict = dict()
                    # adds ele as a key and ele*ele as a value
                    for ele in range(1, n + 1):
                        newDict[ele] = ele * ele
                    print("Dict in the form (x, x*x)", newDict)
                if choice == 6:
                    key = int(input("enter the key u wanna remove"))
                    # removes specified key from dictionary
                    dict1.pop(key)
                    print("after removing key", key, " dict is:", dict1)

                if choice == 7:
                    sample = [{"V": "S001"}, {"V": "S002"}, {"VI": "S001"}, {"VI": "S005"}, {"VII": "S005"}, {"V": "S009"},
                              {"VIII": "S007"}]
                    print(type(sample))

                    uniqueValues = set(val for dic in sample for val in dic.values())
                    print("Unique Values: ", uniqueValues)

                if choice == 8:
                    str1 = input("enter a string")
                    newDict = utility.User.CreateDict(str1)
                    print(newDict)

                if choice == 9:
                    print("\n{:<8} {:<15}".format('Pos', 'Values'))
                    for key, val in dict1.items():
                        print("{:<8} {:<15}".format(key, val))

                if choice == 10:
                    data = [{'id': 1, 'success': True, 'name': 'Lary'}, {'id': 2, 'success': False, 'name': 'Rabi'},
                            {'id': 3, 'success': True, 'name': 'Alex'}]
                    count = 0
                    for item in data:
                        print(item, type(item))
                        # creates dictionary
                        dict1 = dict()
                        # adds vals of list in dict
                        dict1 = item
                        # adds value of key in var val(i.e T or F)
                        val = dict1['success']
                        # if val is True then increases count by 1
                        if val == True:
                            count += 1
                    print("no.of key['success']=True are", count, "times")

                if choice == 11:
                    list1 = [1, 2, 3]
                    newdict = currentdict = {}
                    for name in list1:
                        currentdict[name] = {}
                        currentdict = currentdict[name]
                    print(newdict)

                if choice == 12:
                        count, checkMultiKeys = utility.User.CheckMultiKeys(dict1)
                        print("no.of keys=", count)
                        if checkMultiKeys == True:
                            print("Multiple keys exists in the dictionary\n")
                        else:
                            print("Multiple keys doesn't exists in the dictionary\n")

                if choice == 13:
                    dict1 = {'Alex': ['subj1', 'subj2', 'subj3'], 'David': ['subj1', 'subj2'], 'shweta': [25]}
                    # creates list
                    list1 = []
                    count = 0
                    listcnt = 0
                    # returns vals in dict
                    for val in dict1.values():
                        print(val, type(val))
                        # checks if val is of list type,if true then increases count by 1
                        if type(list1) == type(val):
                            listcnt += 1
                            # counts no of values in list
                            for vals in val:
                                print("list vals=", vals)
                                count += 1
                    print("\nThere are", listcnt, "lists in dictionary")
                    print("\nnumber of items in a dictionary value that are a list", count)
            else:
                raise ValueError
        except ValueError:
            print("\nPlease give valid input")
            print("Try again.....")