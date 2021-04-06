import pyupbit

class My_Upbit(pyupbit.Upbit):

    def __init__(self, access_key, secret_key, coinid='ZRX'):
        super().__init__(access_key, secret_key)

        self.coin_info = self.select_my_coin(coinid)
        self.coinid = coinid

    def select_my_coin(self, coinid):

        coins = self.get_balances()

        coin_index = 0
        my_coin = ''
        for idx, coin in enumerate(coins):
            if coin['currency'] == coinid:
                coin_index = idx
                my_coin = coin
        return my_coin