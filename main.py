from Coin_Predict import Upbit_Data, Upbit_lstm
import time
import os

img_path = 'save_img'
save_img_path = os.path.join(os.getcwd(),img_path)

def start_train(h5_name,coinid='ZRX',):
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    mylstm = Upbit_lstm.My_Lstm()
    times, data = my_upbit.load_ml_data(col='open', interval='minute10', rows=30000)

    mylstm.train_data_load(data, scale=True, train_mode=True, window_size=24, )

    mylstm.train_lstm(loss='mse', metrics='mse', epochs=100, callbacks='go', )

    mylstm.save_md(h5_name)

    pred1 = mylstm.predict(times, ranges=[19000, 20000])

    mylstm.predict_plot(pred1)


def predict_coin(coinid,h5_name='ZRX_winsize24_epoch100.h5',window_size=24):
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    times, data = my_upbit.load_ml_data(col='open', interval='minute10', rows=200) # 10분봉 30000개 데이터수집

    mylstm2 = Upbit_lstm.My_Lstm()

    mylstm2.model_load(h5_name, compile=False)

    mylstm2.train_data_load(data, scale=True, train_mode=False,window_size=window_size)

    mylstm2.predict_last_few(5)

    mylstm2.plot_few_(coinid)

if __name__ == '__main__':
    print('Selete "train":0 or "predict":1 ')
    result = int(input())

    print('input Coin name : ')
    coinid = str(input())

    if result == 0:
        print('input save to h5_name : ')
        h5_name = str(input())

        start_train(h5_name,coinid)

    elif result == 1:
        while True:
            predict_coin(coinid)
            time.sleep(602)
    else:
        print('Done!')

