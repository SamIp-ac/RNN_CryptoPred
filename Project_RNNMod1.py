import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import time


def generate_dataset(price, seq_len):

    X_list, y_list = [], []

    for i in range(len(price) - seq_len):

        X = np.array(price[i:i + seq_len])
        y = np.array(price[i + seq_len])
        X_list.append(X)
        y_list.append(y)

    return np.array(X_list), np.array(y_list)


class RNN_model:

    def build_model(self, model_type='SimpleRNN', seq_len=5):
        model = tf.keras.models.Sequential()
        if model_type == 'SimpleRNN':
            model.add(tf.keras.layers.SimpleRNN(seq_len, activation=tf.nn.relu))
        elif model_type == 'LSTM':
            model.add(tf.keras.layers.LSTM(seq_len, activation=tf.nn.relu))
        elif model_type == 'GRU':
            model.add(tf.keras.layers.GRU(seq_len, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(3, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
        model.compile(optimizer=optimizer, loss='mse')
        return (model)

    def train(self, X_train, y_train, bs=32, ntry=2, type='SimpleRNN', seq_len=5):

        tf.keras.backend.clear_session()

        model = self.build_model(model_type=type, seq_len=seq_len)
        model.fit(X_train, y_train, batch_size=bs, epochs=100, shuffle=True, verbose=0)
        self.best_model = model
        best_loss = model.evaluate(X_train[-50:], y_train[-50:])

        for i in range(ntry):

            tf.keras.backend.clear_session()

            model = self.build_model(model_type=type, seq_len=seq_len)
            model.fit(X_train, y_train, batch_size=bs, epochs=100, shuffle=True, verbose=0)

            if model.evaluate(X_train, y_train) < best_loss:

                self.best_model = model
                best_loss = model.evaluate(X_train[-50:], y_train[-50:])

    def predict(self, X_test):

        return (self.best_model.predict(X_test))

    def summary(self):

        return (self.best_model.summary())

    def evaluate(self, X_test, y_test):

        model = self.best_model
        evaluate = model.evaluate(X_test, y_test)

        return (evaluate)


cwd = os.path.join(os.getcwd(), 'Crypto_Data')
os.chdir(cwd)

STOCK = ['BTC-USD_2y', 'BTC-USD_3m']
train_len = 111  # total row of data - 13 (720 for long/ 111 for short)
seq_len = 3  # 7 vs 3
model_type = ['SimpleRNN', 'LSTM', 'GRU']

for stock in STOCK:

    tf.random.set_seed(4012)

    df = pd.read_csv(f'{stock}.csv')

    df1 = np.array(df['Adj Close']).reshape(-1, 1)

    df1 = pd.DataFrame(df1)
    stock_train = df1.iloc[:train_len].values
    stock_test = df1.iloc[train_len:].values

    X_train, y_train = generate_dataset(stock_train, seq_len=seq_len)
    X_test, y_test = generate_dataset(stock_test, seq_len=seq_len)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    RNN = RNN_model()
    time_start = time.time()
    RNN.train(X_train, y_train, type=model_type[0])  # change Model yourself
    time_used = time.time() - time_start

    y_pred = np.squeeze(RNN.predict(X_test))

    test_len = len(y_test)
    plt.title(f'{stock} with {model_type[0]} model')  # change Model yourself
    plt.plot(range(test_len), y_test, label='True')
    plt.plot(range(test_len), y_pred, label='Prediction')
    plt.legend(['True', 'Prediction'])

    print(RNN.summary())

    acc = mean_squared_error(y_test, y_pred)
    print(f'The mean square error of {model_type[0]} is {acc}')  # change Model yourself
    print(f'The time use {time_used}')
    del RNN
    # plt.show()
