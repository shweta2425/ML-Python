from array import *
import itertools
from itertools import permutations
from copy import deepcopy
import re
import textwrap

class User:
    def __int__(self, arr1):
        self.arr1 = arr1

    def acceptnum(self):
        n = int(input("enter num"))
        return n

    def accepts(self):
        arr1 = array('i', [])
        n = int(input("enter no of elements"))
        for i in range(n):
            ele = int(input("enter element"))
            # adds entered ele into the array
            arr1.append(ele)
        return arr1

    def CheckInt(n):
        try:
            int(n)
            return True
        except Exception:
            return False

    def Diff(lst1, lst2):
        # returns diff of 2 sets & convert is into the list
        diff = list(set(lst1) - set(lst2))
        return diff

    def AppendList(lst1, lst2):
        # adds element at the end of list
        lst1.append(lst2)
        return lst1

    def CommonList(lst1, lst2):
        lst3 = [val for val in lst1 if val in lst2]
        return lst3

    def RemoveDupl(lst):
        # sorts the list in asc order
        lst.sort()
        newlst = list(lst for lst, _ in itertools.groupby(lst))
        return newlst

    def CountFirstLast(lst):
        counter = 0
        for x in lst:
            if len(x) > 2 and x[0] == x[-1]:
                counter += 1
        return counter

    def Compare(lst1, lst2):
        flag = False
        for i in lst1:
            for x in lst2:
                if i == x:
                    flag = True
                    return flag

    def FindCombi(lst):
        # returns permutation of lst
        combi = permutations(lst)
        return combi

    def CalLength(size, lst):
        # creates list
        newlst = []
        for str1 in lst:
            word = str1.split()
            for single in word:
                if len(single) > size:
                    newlst.append(single)
        return newlst



    # Sets functions

    def CreateSet():
        flag = True
        # creates set
        set2 = set()
        while flag:
            ch = input("Wanna add val ?y/n")
            if ch == 'y' or ch == 'Y':
                val = input("enter value")
                # adds val into the set
                set2.add(val)
            if ch == 'n' or ch == 'N':
                flag = False
        return set2

    def DelElement(ele, set1):
        # removes specified element from set & if doesn't find raises an error
        set1.remove(ele)
        return set1

    def DelElementDis(ele, set1):
        # removes specified element from set & if doesn't find still don't raise any error
        set1.discard(ele)
        return set1

    def CreateSet2():
        flag = True
        # creates set
        set2 = set()
        while flag:
            ch = input("Wanna add val ?y/n")
            if ch == 'y' or ch == 'Y':
                val = input("enter value")
                # adds val into the set
                set2.add(val)
            if ch == 'n' or ch == 'N':
                flag = False
        return set2

    def IntersectionSet(set1, set2):
        # return intersection of 2 sets
        newset = set1 & set2
        return newset

    def UnionSet(set1, set2):
        # return union of 2 sets
        newset = set1 | set2
        return newset

    def DifferenceSet(set1, set2):
        # returns difference between 2 sets
        newset = set1 - set2
        return newset

    def SymmetricDifferenceSet(set1, set2):
        # return symmetric diff between 2 sets
        newset1 = set1.symmetric_difference(set2)
        newset2 = set2.symmetric_difference(set1)
        return newset1, newset2

    def ClearSet(set1):
        # empty or clears the set
        set1.clear()
        return set1

    def Findmaxmin(set1):
        # return min value in the set
        small = min(set1)
        # return max value in the set
        maxi = max(set1)
        return small, maxi

    # tuple functions

    def CreateTuple(size):
        tuple2 = tuple()
        for i in range(size):
            val = input("enter value")
            # converting the tuple to list bcz tuple can't be modify once its created
            list1 = list(tuple2)
            # adding ele in list
            list1.append(val)
            # converting list back into tuple
            tuple2 = tuple(list1)
        return tuple2

    def RemoveTitem(val, tuple1):
        list1 = []
        # converting tuple into list
        list1 = list(tuple1)
        # removing specified element
        list1.remove(val)
        # converting list back into tuple
        tuple1 = tuple(list1)
        return tuple1

    def ConvertToTuple(size, list1):
        # converting list into tuple
        tuple1 = tuple(list1)
        return tuple1

    def Slice(start, end, tuple1):
        # slicing
        return tuple1[start:end]

    def Clone(tuple1):
        # creates copy of tuple
        cloneTuple = deepcopy(tuple1)
        return cloneTuple

    # Dictionary functions

    def SortDict(dict1):
        # sorts dict is asc order
        asc = sorted(dict1.values())
        # sorts dic in desc order
        desc = sorted(dict1.values(), reverse=True)
        return asc, desc

    def CreateDict(str1):
        # creates dictionary
        newDict = dict()

        for char in str1:
            # adding characters of string as a key & counting occurrence of char in dictionary
            newDict[char] = str1.count(char)
        return newDict

    def CheckMultiKeys(dict1):
        count = 0
        for keys in dict1:
            # counts no.of keys in dict
            count += 1
        #     if count is greater than one means multiple keys exists in dict
        if count > 1:
            return count, True

    # String Functions

    def CheckLen(string):
        # return length of string
        length = len(string)
        return length

    def CheckString(string):
        # validate for space
        regex =re.compile('[+\s+]')
        # check if input is string (space also allowed)
        if string.isalpha() or regex.search(string):
            return True

    # special chars also accepted
    def CheckStr1(string):
        # validate for special chars
        regex = re.compile('[.@_!#$%^&*()<>?/\|}{~:+\s+]')
        # check if input is string (space also allowed)
        if string.isalpha() or regex.search(string):
            return True

    def ReplaceStr(string):
        # stores first char of string in var char
        char = string[0]
        # replaces that char with $
        str1 = string.replace(char, '$')
        # concatenating string except first char
        str1 = char + str1[1:]
        return str1

    def AddIng(str1):
        str2 = 'ing'
        str3 = 'ly'
        # if str is greater than 2 and doesn't ends with ing then adds ing at the end of str
        if len(str1) > 2 and str1.find(str2) == -1:
            newStr = str1 + str2
        # if str is greater than 2 and ends with ing then adds ly at the end of str
        elif len(str1) > 2 and str1.find(str2) != -1:
            newStr = str1.replace('ing', 'ly')
        else:
            # if str is less than 3 chars then returns as it is
            return str1
        return newStr

    def CreateList(size):
        # creates list
        lst = []
        for i in range(size):
            words = input("enter words")
            # adds words in list
            lst.append(words)
        return lst

    def FindMaxWord(lst):
        # stores 1st word's length as a max
        max = len(lst[0])
        for word in lst:
            # calculate length of word
            length = len(word)
            # if another word's len is max then stores it as max
            if max < length:
                max = length
        return max

    def ConvertStr(str1):
        # convert string into upper case
        upper = str1.upper()
        # convert string into lower case
        lower = str1.lower()
        return upper, lower

    def ReverseStr(str1):
        # reverse a string
        return str1[::-1]

    def ConvertfirstNtoLower(str1,numofChars):
        # string slicing
        substr=str1[:numofChars]
        # print(substr)
        # convert string into lowercase
        substrinLowercase = substr.lower()
        return substrinLowercase

    def CountOccurrence(string,substr):
        # return count of sub string in the string
        count1=string.count(substr)
        return count1

    def CheckStrsubstr(string,substr):
        # validate for special chars
        regex = re.compile('[.@_!#$%^&*()<>?/\|}{~:\w+\s\w+]')
        # checks if str & sub str is a string(space also allowed or spl chars)
        if string.isalpha() or regex.search(string) or substr.isalpha():
            return True
    def Format(str1):
        wrapper = textwrap.TextWrapper(width=50)
        string = wrapper.fill(text=str1)
        return string

    def Rsplit(str1):
        # split str into list starting from right,in 2(i.e 0 &1 )parts
        string=str1.rsplit(" ",1)
        return string