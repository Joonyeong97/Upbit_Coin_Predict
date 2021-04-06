from Coin_Predict import Upbit_Data, Upbit_lstm

def start_train(h5_name,coinid='ZRX',):
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    mylstm = Upbit_lstm.My_Lstm()
    times, data = my_upbit.load_ml_data(col='open', interval='minute10', rows=30000)

    mylstm.train_data_load(data, scale=True, train_mode=True, window_size=24, )

    mylstm.train_lstm(loss='mse', metrics='mse', epochs=100, callbacks='go', )

    mylstm.save_md(h5_name)

    return times, data

def predict_coin(coinid,h5_name='ZRX_winsize24_epoch100.h5',window_size=24):
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    times, data = my_upbit.load_ml_data(col='open', interval='minute10', rows=200)

    mylstm2 = Upbit_lstm.My_Lstm()

    mylstm2.model_load(h5_name, compile=False)

    mylstm2.train_data_load(data, scale=True, train_mode=False,window_size=24)

    mylstm2.predict_last_few(5)

    mylstm2.plot_few_()

if __name__ == '__main__':
    print('selete "train" or "predict" : ')
    result = str(input())

    print('input Coin name : ')
    coinid = str(input())

    if result == 'train':
        print('input save to h5_name : ')
        h5_name = str(input())

        start_train(h5_name,coinid)
    elif result == 'predict':
        predict_coin(coinid)
    else:
        print('Done!')

