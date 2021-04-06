import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


plt.rcParams['font.family'] = 'Malgun Gothic'

class My_Lstm:
    def windowed_dataset(self, series, window_size, batch_size, shuffle_buffer):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.map(lambda w: (w[:-1], w[1:]))

        return ds.batch(batch_size).prefetch(1)

    def __dataframe_to_series(self, dataframe, col='open'):
        data = dataframe

        return np.asarray(data[col], dtype='float32')

    def scale_data_fit(self, series):
        self.min_scale = np.min(series)
        self.max_scale = np.max(series)

    def scale_data(self, series):
        series = np.array(series)
        series -= self.min_scale
        series /= self.max_scale

        return series

    def un_scale_data(self, series):
        series = np.array(series)
        series *= self.max_scale
        series += self.min_scale

        return series

    def keras_layers_compile(self, loss='mae', optimizer='adam', metrics='mae'):
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=[None, 1], return_sequences=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(1)])

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def callbacks(self, monitor='loss', mode='min', patience=10):
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 10))
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='mse', patience=10)
        return [lr_schedule, earlystop]

    def fit_lstm(self, dataset, epochs=100, callbacks=None):
        if callbacks:
            history = self.model.fit(dataset, epochs=epochs, callbacks=callbacks)
        else:
            history = self.model.fit(dataset, epochs=epochs)
        return history

    def train_data_load(self, dataframe, col='open', scale=True, train_mode=True, window_size=24, batch_size=16,
                        shuffle_buffer=30000):
        self.scale = scale

        # Dataframe to series 하나의 컬럼만 가져와서 예측진행
        series = self.__dataframe_to_series(dataframe, col=col)
        self.times = dataframe.index

        # 원본데이터 백업
        self.backup_series = series

        # Scale 진행 MinMax scale

        if train_mode:
            if self.scale:
                self.scale_data_fit(series)
                series = self.scale_data(series)
        else:
            if self.scale:
                self.scale_data_fit(series)

        # Tensorflow 전용 데이터셋으로 변환 (window size=예측할 날짜수)
        self.window_size = window_size

        if train_mode:
            series = self.windowed_dataset(series, self.window_size, batch_size, shuffle_buffer)

            self.series = series
        else:
            self.series = series

    def train_lstm(self, epochs=100, callbacks=None, loss='mae', optimizer='adam', metrics='mae'):

        # Keras
        self.keras_layers_compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if callbacks:
            callbacks = self.callbacks()
        else:

            pass
        history = self.fit_lstm(self.series, epochs=epochs, callbacks=callbacks)

        return history

    def save_md(self, name):
        try:
            self.model.save(name)
        except:
            print('Error!')

    def model_load(self, name, compile=True):
        if compile:
            self.keras_layers_compile()
            self.model = tf.keras.models.load_model(name)
        else:
            self.model = tf.keras.models.load_model(name)

    def predict(self, times, ranges=[0, 200]):
        predicts = []
        self.ranges = ranges
        # self.backup_series_to_un_scale = self.un_scale_data(self.backup_series[ranges[0]:ranges[1]])

        self.time_series = times[ranges[0]:ranges[1]]

        ranges_series = self.backup_series[ranges[0]:ranges[1]]

        if self.scale:

            ranges_series_scale = self.scale_data(ranges_series)

            for time in range(len(ranges_series_scale) - self.window_size):
                pred = np.array(ranges_series_scale[time: time + self.window_size])

                pred = pred.reshape(1, -1, 1)

                predict = self.model.predict(pred)

                predicts.append(predict[0][0])
            # 수정부분
            predicts = self.un_scale_data(predicts)
        else:

            for time in range(len(ranges_series) - self.window_size):
                pred = np.array(ranges_series[time: time + self.window_size])

                pred = pred.reshape(1, -1, 1)

                predict = self.model.predict(pred)

                predicts.append(predict[0][0])

        return predicts

    def predict_plot(self, predicts):

        plt.figure(figsize=(12, 8))
        plt.plot(self.time_series[:-self.window_size], predicts, color='red', label='Predict')
        plt.plot(self.time_series[:-self.window_size],
                 self.backup_series[self.ranges[0]:self.ranges[1] - self.window_size], color='blue', label='Real')
        plt.legend(loc='center left')
        plt.show()

    def predict_last_few(self, future_):

        _last_future_series = self.backup_series[-self.window_size:]

        self.pred_times = times[-self.window_size:]
        self.pred_times = self.pred_times.to_list()

        self.future_ = future_
        futures = []

        if self.scale:

            _last_future_series = self.scale_data(_last_future_series)

            for time in range(future_):
                pred = np.array(_last_future_series, dtype='float64')

                pred = pred.reshape(1, -1, 1)

                predict = self.model.predict(pred)

                _last_future_series = np.append(_last_future_series, predict[0][0])

                futures.append(predict[0][0])

                # 시간도 동일하게 추가 및 제거
                self.pred_times.append(self.pred_times[-1] + timedelta(minutes=10))  # 추후 변경예정
                self.pred_times.pop(0)

                _last_future_series = np.delete(_last_future_series, 0)

            futures = self.un_scale_data(futures)
            _last_future_series = self.un_scale_data(_last_future_series)

        else:

            for time in range(future_):
                pred = np.array(_last_future_series, dtype='float64')

                pred = pred.reshape(1, -1, 1)

                predict = self.model.predict(pred)

                _last_future_series = np.append(_last_future_series, predict[0][0])
                futures.append(predict[0][0])

                # 시간도 동일하게 추가 및 제거
                self.pred_times.append(self.pred_times[-1] + timedelta(minutes=10))  # 추후 변경예정
                self.pred_times.pop(0)

                _last_future_series = np.delete(_last_future_series, 0)

        self.futures = futures
        self._last_future_series = _last_future_series

        self.pred_times = pd.DatetimeIndex(self.pred_times)

    def plot_few_(self, coinid):
        threshold = np.ones_like(self._last_future_series, dtype=bool)
        threshold[:-self.future_] = False

        pred_y = self._last_future_series

        plt.plot(self.pred_times, pred_y, color='blue', label='Real')
        plt.plot(self.pred_times[threshold], pred_y[threshold], color='red', label='Real')
        plt.title(f"{coinid}  {self.future_}개 예측결과")
        plt.show()