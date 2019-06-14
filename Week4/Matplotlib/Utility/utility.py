
class UtilityClass:

    def accept_size(self):
        n = int(input("enter how many values u wanna plot"))
        return n

    def CreateList(self, size):
        lst = []
        for i in range(size):
            words = int(input("enter value"))
            lst.append(words)
        return lst

    def accept_languages(self, size):
        lst = []
        for i in range(size):
            words = input("enter language name")
            lst.append(words)
        return lst

    def accept_popularity(self, size):
        lst = []
        for i in range(size):
            words = float(input("enter value"))
            lst.append(words)
        return lst

    def accept_position(self, size):
        lst = []
        for i in range(size):
            position = float(input("enter position"))
            lst.append(position)
        return lst

    def accept_width(self, size):
        lst = []
        for i in range(size):
            width = float(input("enter width"))
            lst.append(width)
        return lst

    def CheckInt(self, val):
        try:
            int(val)
            return True
        except Exception:
            return False
