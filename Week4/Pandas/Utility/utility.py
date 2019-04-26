import numpy as np
import pandas as pd
class User:
    def __init__(self):
        self.a=10

    def creates_series(self, arr):
        series = pd.Series(arr)
        return series
