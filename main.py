from Coin_Predict import Upbit_Data, Upbit_lstm
import time
import os

img_path = 'save_img'
save_img_path = os.path.join(os.getcwd(),img_path)
# 코인이름
coinid = 'ZRX'
# 분봉
minutes = 10
interval = 'minute10'
# 예측할 갯수
predict_count = 3

# window size
window_size = 12

# 가중치 파일이름
h5_file_name = 'ZRX_winsize24_epoch100.h5'



def start_train(h5_name,coinid='ZRX',window_size=12):
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    mylstm = Upbit_lstm.My_Lstm()
    times, data = my_upbit.load_ml_data(col='open', interval='minute10', rows=30000)

    mylstm.train_data_load(data, scale=True, train_mode=True, window_size=window_size, )

    mylstm.train_lstm(loss='mse', metrics='mse', epochs=100, callbacks='go', )

    mylstm.save_md(h5_name)

    pred1 = mylstm.predict(times, ranges=[19000, 20000])

    mylstm.predict_plot(pred1)


def predict_coin(coinid,h5_file_name=h5_file_name,window_size=24,interval='minute1',predict_count=5,minutes=10):
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    times, data = my_upbit.load_ml_data(col='open', interval=interval, rows=200) # 10분봉 30000개 데이터수집

    mylstm2 = Upbit_lstm.My_Lstm()

    mylstm2.model_load(h5_file_name, compile=False)

    mylstm2.train_data_load(data, scale=True, train_mode=False,window_size=window_size,interval=interval)

    mylstm2.predict_last_few(predict_count,minutes=minutes)

    mylstm2.plot_few_(coinid)

if __name__ == '__main__':
    print('Selete "train":0 or "predict":1 ')
    result = int(input())

    # print('input Coin name : ')
    # coinid = str(input())

    if result == 0:
        print('input save to h5_name : ')
        h5_name = str(input())

        start_train(h5_name,coinid)

    elif result == 1:
        while True:
            predict_coin(coinid,window_size=window_size,interval=interval,predict_count=predict_count,minutes=minutes)
            time.sleep(601)
    else:
        print('Done!')

