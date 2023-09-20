#!/usr/bin/env python3

"""Manipulating Time Series Data."""

from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from projectpro import model_snapshot
import pandas as pd
import pandas_ta as pdta
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


SCALE = MinMaxScaler(feature_range=(0,1))


# Loading the data using Yahoo Finance API.
def load_data(stock_name, start_date, end_date):
    """
        Loads the data from yahoo finance.
    """
    dataset = yf.download(stock_name, start=start_date, end=end_date)
    return dataset


# This is a function just to visulise the data.
def train_test_plot(dataset, tstart, tend):
    """
        Plots the data for a better visualisation.
    """
    dataset.loc[f"{tstart}":f"{tend}", "High"].plot(figsize=(16, 4), legend=True)
    dataset.loc[f"{tend+1}":, "High"].plot(figsize=(16, 4), legend=True)
    plt.legend([f"Train (Before {tend+1})", f"Test ({tend+1}) and beyond)"])
    plt.show()


# Creating train and test datasets.
def test_train_split(dataset, start, end, columns = ['High']):
    """
        Splits the dataset into a training dataset and a testing dataset.
    """
    train = dataset.loc[f"{start}":f"{end}", columns].values
    test = dataset.loc[f"{end+1}":, columns].values
    return train, test


# Scaling dataset values.
def scale_train_data(dataset):
    """
        Normalises the data for a faster learning model.
    """
    dataset = dataset.reshape(-1, 1)
    dataset_scaled = SCALE.fit_transform(dataset)
    return dataset_scaled


def split_sequence(sequence, n_steps):
    """
        Makes the data in an overlapping form for some sort of training.
    """
    x_list, y_list = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x_list.append(seq_x)
        y_list.append(seq_y)

    return np.array(x_list), np.array(y_list)


def input_output(dataset, features):
    """
        Trains the model using the training dataset.
    """
    scaled_set = scale_train_data(dataset)
    input_set, label = split_sequence(scaled_set, 1)
    input_set = input_set.reshape(input_set.shape[0], input_set.shape[1], features)
    return input_set, label



def plot_predictions(test, predicted, title):
    """
        Plots the predicted values for running the model on the testing_set.
    """
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title(f'{title}')
    plt.xlabel("Time")
    plt.ylabel(f'{title}')
    plt.legend()
    plt.show()


def print_rmse(test, predicted):
    """
        Calculates the mean squared error for our model.
    """
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print(f"The root mean squared error is {rmse}")


def create_rnn_model():
    """
        Creates the model that will predict the evolution of our stock.
        We are using RNN.
    """
    model_rnn = tf.keras.models.Sequential()
    n_steps = 1
    features = 1
    model_rnn.add(tf.keras.layers.SimpleRNN(units=125, input_shape=(n_steps, features)))
    model_rnn.add(tf.keras.layers.Dense(units=1))
    return model_rnn


def create_lstm_model():
    """
        Creates an lstm model to predict the evolution of our stock.
    """
    model_lstm = tf.keras.models.Sequential()
    n_steps = 1
    features = 1
    model_lstm.add(tf.keras.layers.LSTM(units=125, input_shape=(n_steps, features)))
    model_lstm.add(tf.keras.layers.Dense(units=1))
    return model_lstm


def plot_loss(history):
    """
        Plots the loss.
    """
    plt.figure(figsize=(15, 10))
    plt.plot(history.history['loss'], label='loss')
    plt.legend(loc='best')
    plt.show()


def train_model(model, training_set, features):
    """
        Trains the model.
    """
    x_train, label = input_output(training_set, features)
    history = model.fit(x_train, label, epochs=10, batch_size=32, verbose=0)
    model_snapshot("34db30")
    # Plotting the loss.
    plot_loss(history)


def test_model(model, testing_set):
    """
        Tests the model.
    """
    scaled_test = SCALE.transform(testing_set.reshape(-1,1))
    x_test, y_test = split_sequence(scaled_test, 1)
    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = SCALE.inverse_transform(predicted_stock_price)
    # Plotting the predictions
    plot_predictions(testing_set, predicted_stock_price, "Apple Stock Price")


def main():
    """
        The main function of the program.
    """
    features = 1
    n_steps = 1
    tstart = 2016
    tend = 2020
    dataset = load_data('AAPL', datetime(2012, 1, 1), datetime.now())
    training_set, testing_set = test_train_split(dataset, tstart, tend)
    #model = create_rnn_model()
    # Compiling the model.
    #model.compile(optimizer="RMSprop", loss="mse")
    #train_model(model, training_set, features)
    # Prediction -- Testing.
    #test_model(model, testing_set)

    def sequence_generation(dataset: pd.DataFrame,
                            scale: MinMaxScaler,
                            model: tf.keras.models.Sequential,
                            steps_future: int):
        high_dataset = dataset.iloc[len(dataset) - len(testing_set) - n_steps:]["High"]

        high_dataset = scale.transform(high_dataset.values.reshape(-1, 1))
        inputs = high_dataset[:n_steps]

        for _ in range(steps_future):
            curr_pred = model.predict(inputs[-n_steps:].reshape(-1, n_steps, features), verbose=0)
            inputs = np.append(inputs, curr_pred, axis=0)

        return scale.inverse_transform(inputs[n_steps:])

    steps_in_future = 25
    #results = sequence_generation(dataset, SCALE, model, steps_in_future)
    #plot_predictions(testing_set[:steps_in_future], results, "Apple Stock Price")
    print()
    print("******** END OF RNN *********")
    print()
    print("*********** LSTM ************")
    model_lstm = create_lstm_model()
    model_lstm.compile(optimizer="RMSprop", loss="mse")
    train_model(model_lstm, training_set, features)
    test_model(model_lstm, testing_set)

    results = sequence_generation(dataset, SCALE, model_lstm, steps_in_future)
    plot_predictions(testing_set[:steps_in_future], results, "Apple Stock Price")

    # Multi-variate
    mv_features = 6
    multi_variate_df = dataset.copy()
    # Creating technical indexes
    multi_variate_df['RSI'] = pdta.rsi(multi_variate_df.Close, length=15)
    multi_variate_df['EMAF'] = pdta.ema(multi_variate_df.Close, length=20)
    multi_variate_df['EMAM'] = pdta.ema(multi_variate_df.Close, length=100)
    multi_variate_df['EMAS'] = pdta.ema(multi_variate_df.Close, length=150)
    # Creating labels
    multi_variate_df['Target'] = multi_variate_df['Adj Close'] - dataset.Open
    multi_variate_df['Target'] = multi_variate_df['Target'].shift(-1)
    multi_variate_df.dropna(inplace=True)
    multi_variate_df.drop(['Volume', 'Close'], axis=1, inplace=True)

    multi_variate_df.loc[f"{tstart}":f"{tend}", ['High', 'RSI']].plot(figsize=(16, 4), legend=True)
    plt.title("Apple RSI")
    plt.show()

    multi_variate_df.loc[f"{tstart}": f"{tend}", ['High', 'EMAF', 'EMAM', 'EMAS']].plot(figsize=(16, 4), legend=True)
    plt.title("Apple Stock Averages")
    plt.show()

    feat_columns = ['Open', 'High', 'RSI', 'EMAF', 'EMAM', 'EMAS']
    label_col = ['Target']

    mv_training_set, mv_testing_set = test_train_split(multi_variate_df, tstart, tend, feat_columns + label_col)
    x_train = mv_training_set[:, :-1]
    y_train = mv_training_set[:, -1]

    x_test = mv_testing_set[:, :-1]
    y_test = mv_testing_set[:, -1]
    # Scaling
    mv_scale = MinMaxScaler(feature_range=(0, 1))
    x_train = mv_scale.fit_transform(x_train).reshape(-1, 1, mv_features)
    x_test = mv_scale.transform(x_test).reshape(-1, 1, mv_features)
    # Model
    model_mv = tf.keras.models.Sequential()
    model_mv.add(tf.keras.layers.LSTM(units=125, input_shape=(1, mv_features)))
    model_mv.add(tf.keras.layers.Dense(units=1))

    model_mv.compile(optimizer="RMSprop", loss="mse")

    history = model_mv.fit(x_train, y_train, epochs=20, batch_size=32, verbose=False)
    model_snapshot("34db30")
    plot_loss(history)

    predictions = model_mv.predict(x_test)
    plot_predictions(y_test, predictions, "Apple Stock Price!")



if __name__ == "__main__":
    main()
