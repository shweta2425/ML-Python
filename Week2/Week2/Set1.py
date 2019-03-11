# 1.Write a Python program to create a set.
# 2.Write a Python program to iteration over sets.
# 3.Write a Python program to add member(s) in a set.
# 4. Write a Python program to remove item(s) from set
# 5.Write a Python program to remove an item from a set if it is present in the set
# 6.Write a Python program to create an intersection of sets
# 7.Write a Python program to create a union of sets.
# 8.Write a Python program to create set difference
# 9.Write a Python program to clear a set.
# 10.Write a Python program to use of frozensets.
# 11.Write a Python program to find maximum and the minimum value in a set

# set1={"shweta","pankaj","shadrach","ethel","danish"}
# print(set1)
import os
from Week2.Utilities import utility


class Set1:
    flag=True

    # obj = utility.User.CreateSet()
    # print("created set is", obj)

    while flag:
        print("1.Create set")
        print("2.Iterate over set")
        print("3.Remove Item from set")
        print("4.Remove Item from set using discard()")
        print("5.Find Intersection of sets")
        print("6.Find Union of sets")
        print("7.Find Difference of sets")
        print("8.Find Symmetric Difference of sets")
        print("9.Clear a set.")
        print("10.Use of Frozenset()")
        print("11.find max & min value from set")
        print("0.exit")
        try:
            choice = int(input("Enter ur choice"))
            if choice >= 0 and choice <= 11:

                if choice == 0:
                    flag = False
                    os.system('clear')

                if choice == 1:
                    set1 = utility.User.CreateSet()
                    print("created set is", set1)

                if choice == 2:
                    for ele in set1:
                        print(ele)

                if choice == 3:
                    delEle = input("Enter element to remove")
                    setaftrRemoving = utility.User.DelElement(delEle, set1)
                    print("set after removing :",setaftrRemoving)

                if choice == 4:
                    delEle = input("Enter element to remove")
                    setaftrRemoving = utility.User.DelElementDis(delEle, set1)
                    print("set after removing :", setaftrRemoving)

                if choice == 5:
                    print("Create 1st set")
                    set1 = utility.User.CreateSet()
                    print("created set1 is", set1)

                    print("Create 2nd set")
                    set2 = utility.User.CreateSet2()
                    print("created set2 is", set2)

                    Intersectionset=utility.User.IntersectionSet(set1, set2)
                    print("set1",set1)
                    print("set2", set2)

                    print("Intersection of 2 sets is :", Intersectionset)

                if choice == 6:
                    print("Create 1st set")
                    set1 = utility.User.CreateSet()
                    print("created set1 is", set1)

                    print("Create 2nd set")
                    set2 = utility.User.CreateSet2()
                    print("created set2 is", set2)

                    Unionset = utility.User.UnionSet(set1, set2)
                    print("set1", set1)
                    print("set2", set2)

                    print("Union of 2 sets is :", Unionset)

                if choice == 7:
                    print("Create 1st set")
                    set1 = utility.User.CreateSet()
                    print("created set1 is", set1)

                    print("Create 2nd set")
                    set2 = utility.User.CreateSet2()
                    print("created set2 is", set2)

                    Diffset = utility.User.DifferenceSet(set1, set2)
                    print("set1", set1)
                    print("set2", set2)

                    print("Difference (set1-set2) is :", Diffset)

                if choice == 8:
                    print("Create 1st set")
                    set1 = utility.User.CreateSet()
                    print("created set1 is", set1)

                    print("Create 2nd set")
                    set2 = utility.User.CreateSet2()
                    print("created set2 is", set2)

                    diffbtw12, diffbtw21 = utility.User.SymmetricDifferenceSet(set1, set2)
                    print("set1", set1)
                    print("set2", set2)

                    print("Symmetric Difference between Set1 & Set2", diffbtw12)
                    print("Symmetric Difference between Set2 & Set1", diffbtw21)

                if choice == 9:
                    # print("inside 9 opt")
                    set2 = utility.User.CreateSet2()
                    print("created set is", set2)
                    choice1 = input("wanna clear the set ?y/n")
                    flg=False
                    while flg != True:
                        if choice1 == 'y' or choice1 == 'Y':
                            ClearedSet = utility.User.ClearSet(set2)
                            flg=True

                        if choice1 == 'n' or choice1 == 'N':
                            flg = False
                    print(ClearedSet)

                if choice == 10:
                    person = {"name": "John", "age": 23, "sex": "male"}
                    # returns an immutable frozenset & can be used as key in dictionary
                    fSet = frozenset(person)
                    print('The frozen set is:', fSet)

                if choice == 11:
                    print("Create 1st set")
                    set1 = utility.User.CreateSet()
                    print("created set1 is", set1)
                    min1, max2 =utility.User.Findmaxmin(set1)
                    print("small :", min1)
                    print("max :", max2)

            else:
                raise ValueError
        except ValueError:
            print("\nPlease give valid input")
            print("Try again.....")