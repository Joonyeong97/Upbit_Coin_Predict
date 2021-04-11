import pyupbit
import pandas as pd
import time
import numpy as np


class My_Upbit_Data():

    def __init__(self, coinid='ZRX'):
        self.coinid = coinid

    def __load_data(self, interval='minute10', date=None, rows=1000):
        dfs = []

        ranges = int(rows / 200)

        for i in range(ranges):
            df = pyupbit.get_ohlcv(ticker='KRW-' + self.coinid, interval=interval, to=date)
            dfs.append(df)

            date = df.index[0]
            time.sleep(0.15)

        df = pd.concat(dfs).sort_index()

        return df

    def load_ml_data(self, interval='minute10', rows=20000):
        dataframe = self.__load_data(interval=interval, rows=rows)
        times = dataframe.index

        return [times, dataframe]