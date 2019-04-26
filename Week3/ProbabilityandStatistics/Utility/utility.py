class User:

    def calProbability(self, possiblities, totalOutcome):
        return possiblities / totalOutcome

    def oneHead(self, SampleList):
        oneH = 0
        for item in SampleList:
            cnt = 0
            for char in item:
                if char == 'H':
                    cnt += 1

            if cnt == 1:
                oneH += 1
        return oneH

    def FindTwoHead(self, SampleList):
        twoHead = 0
        for item in SampleList:
            cnt = 0
            for char in item:
                if char == 'H':
                    cnt += 1

            if cnt >= 2:
                twoHead += 1
        return twoHead

    def AtleastOneHead(self, SampleList):
        oneH = 0
        for item in SampleList:
            cnt = 0
            for char in item:
                if char == 'H':
                    cnt += 1

            if cnt >= 1:
                oneH += 1
        return oneH
