# 1.Write a Python program to create a tuple
# 2.Write a Python program to create a tuple with different data types
# 3. Write a Python program to unpack a tuple in several variables.
# 4.Write a Python program to create the colon of a tuple
# 5.Write a Python program to find the repeated items of a tuple
# 6.Write a Python program to check whether an element exists within a tuple
# 7.Write a Python program to convert a list to a tuple
# 8.Write a Python program to remove an item from a tuple.
# 9.Write a Python program to slice a tuple.
# 10.Write a Python program to reverse a tuple.


# tuple1=("Shweta","mohite","25","1993")
# print(tuple1)

from Week2.Utilities import utility


class Tuple1:
    flag = True

    # obj = utility.User.CreateSet()
    # print("created set is", obj)

    while flag:
        print("1.Create Tuple")
        print("2.Unpack Tuple")
        print("3.Find repeated items")
        print("4.Clone a tuple")
        print("5.Check whether item exist in tuple")
        print("6.Convert list into tuple")
        print("7.Remove item")
        print("8.Slicing ")
        print("9.Reverse tuple")
        print("0.Exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice >= 0 and choice <= 9:
                if choice == 0:
                    flag = False

                if choice == 1:
                    size = int(input("Enter no.of values do u wanna add"))
                    tuple1 = utility.User.CreateTuple(size)
                    print("created tuple is", tuple1, type(tuple1))

                if choice == 2:
                    # size = int(input("Enter no.of values do u wanna add"))
                    # obj = utility.User.CreateTuple(size)
                    # print("created tuple is", obj, type(obj),type(obj[0]))
                    #
                    # print("You have created", size, "values so create", size, "variables")
                    #
                    # list1=[]
                    # for i in range(size):
                    #     val = input("enter variable name")
                    #     list1.append(val)
                    #
                    # for i in range(size):
                    #     list1[i] = obj[i]
                    #
                    #
                    # print(list1)
                    # obj1 = utility.User.UnpackTuple(size, obj)
                    x = ("BridgeLabz", 101 , "Software Engineer")  # tuple packing
                    (company, eid, profile) = x  # tuple unpacking
                    print(company, type(company))
                    print(eid, type(eid))
                    print(profile)

                if choice == 3:
                    size = int(input("Enter no.of values do u wanna add"))
                    tuple1 = utility.User.CreateTuple(size)
                    print("created tuple is", tuple1, type(tuple1))
                    list1 = []
                    # converting tuple into list
                    list2 = list(tuple1)

                    for item in list2:
                        # if count of item is greater than one then adding into list as repeated ele
                        if list2.count(item)>1:
                            list1.append(item)
                    print("repeated items are", list1)
                    # to remove duplicates convering it into set
                    set1=set(list1)
                    print("set is", set1)

                if choice == 9:
                    size = int(input("Enter no.of values do u wanna add"))
                    tuple1 = utility.User.CreateTuple(size)
                    print("created tuple is", tuple1, type(tuple1))
                    # reversing tuple elements
                    for element in reversed(tuple1):
                        print(element)

                if choice == 7:
                    size = int(input("Enter no.of values do u wanna add"))
                    tuple1 = utility.User.CreateTuple(size)
                    print("created tuple is", tuple1, type(tuple1))

                    val = input("enter item to remove")
                    tupleafterRemoving = utility.User.RemoveTitem(val, tuple1)
                    print("Tuple after removing item", tupleafterRemoving)

                if choice == 6:
                    list1 = []
                    size = int(input("enter no.of val u wanna add into the list"))
                    for i in range(size):
                        val = input("enter value")
                        # adding val into list
                        list1.append(val)
                    print("list is", list1)

                    tuple1 = utility.User.ConvertToTuple(size, list1)
                    print("new tuple ", tuple1)

                if choice == 5:
                    size = int(input("Enter no.of values do u wanna add"))
                    tuple1 = utility.User.CreateTuple(size)
                    print("created tuple is", tuple1, type(tuple1))
                    val = input("enter value to check in tuple")
                    # checking if val exist in tuple
                    print(val in tuple1)

                if choice == 8:
                    size = int(input("Enter no.of values do u wanna add"))
                    tuple1 = utility.User.CreateTuple(size)
                    print("created tuple is", tuple1, type(tuple1))

                    start = int(input("enter start of slicing"))
                    end = int(input("enter end of slicing"))

                    slice1 = utility.User.Slice(start, end, tuple1)
                    print("sliced tuple is", slice1)

                if choice == 4:
                    size = int(input("Enter no.of values do u wanna add"))
                    tuple1 = utility.User.CreateTuple(size)
                    print("created tuple is", tuple1, type(tuple1))
                    cloneTuple = utility.User.Clone(tuple1)
                    print("Clone tuple: ", cloneTuple, type(cloneTuple))

            else:
                raise ValueError
        except ValueError:
            print("\nPlease give valid input")
            print("Try again.....")