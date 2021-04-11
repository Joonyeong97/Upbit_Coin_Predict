from Coin_Predict import Upbit_Data, Upbit_lstm
import time
import os

# 결과 이미지 저장
img_path = 'save_img'
save_img_path = os.path.join(os.getcwd(),img_path)

##### 예측에 사용되는 변수들 #####
# 코인이름
coinid = 'ZRX'

# 분봉
minutes = 10
interval = 'minute10'

# 예측할 갯수
predict_count = 3

# 로드할 가중치 파일이름
h5_file_name = 'ZRX_winsize12_epoch100_batch12_minute10_2625_349.h5'

# 학습된 scale 값 입력
max_scale = 2625
min_scale = 349

# 시작가,종료가 설정 open or close
col = 'close'

##### 학습에 사용되는 변수들 #####

# 시작가 or 종료가
col = 'close'
# 분봉 단위 및 day단위
interval = 'minute10'
# API를 통해서 가져올 데이터의 수
rows = 50000

# 훈련횟수
epochs = 100
# 예측 1개에 사용할 갯수
window_size = 12
# 훈련시 학습에 사용할 데이터의 집합 크기
batch_size = 12
# random(데이터의 수 보다 커야함)
shuffle_buffer = 60000
# 저장할 h5 이름
save_h5_name = 'train.h5'



def start_train():
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    mylstm = Upbit_lstm.My_Lstm()
    times, data = my_upbit.load_ml_data(interval='minute10', rows=30000)

    mylstm.train_data_load(data, scale=True, train_mode=True, window_size=window_size,batch_size=batch_size,shuffle_buffer=shuffle_buffer,interval=interval)

    mylstm.train_lstm(loss='mse', metrics='mse', epochs=100, callbacks='go',)

    mylstm.save_md(save_h5_name)

    pred1 = mylstm.predict(times, ranges=[-1500, -1])

    mylstm.predict_plot(pred1)


def predict_coin():
    my_upbit = Upbit_Data.My_Upbit_Data(coinid=coinid)
    times, data = my_upbit.load_ml_data(interval=interval, rows=200) # 10분봉 30000개 데이터수집

    mylstm2 = Upbit_lstm.My_Lstm()

    mylstm2.model_load(h5_file_name, compile=False)

    mylstm2.train_data_load(data,col=col, scale=True, train_mode=False,window_size=window_size,interval=interval)

    mylstm2.predict_last_few(predict_count,minutes=minutes)

    mylstm2.plot_few_(coinid)

if __name__ == '__main__':
    print('Selete "train":0 or "predict":1 ')
    result = str(input())

    # print('input Coin name : ')
    # coinid = str(input())

    if result == '0':
        print('input save to h5_name : ')
        h5_name = str(input())

        start_train()

    elif result == '1':
        while True:
            predict_coin()
            time.sleep(600)

    elif result == 'cuda':
        import tensorflow as tf
        from tensorflow.python.client import device_lib

        print(tf.__version__)
        print(device_lib.list_local_devices())

    else:
        print('Done!')

