from abc import ABC
import pandas as pd
import numpy as np
from tensorflow.keras import *
from tensorflow import keras
# 伪代码

from layer.TimeLSTMCell_2 import *


class Decoder(tf.keras.Model):
    def __init__(self, hidden_size, feature_dims, model_type):
        super(Decoder, self).__init__(name='Decoder')
        self.feature_dims = feature_dims
        # 首先定义各种layer层的各部分
        # ENCODER
        self.model_type = model_type
        self.hidden_size = hidden_size
        if model_type == 'LSTM':
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)

        elif model_type == 'TimeLSTM2':
            self.LSTM_decoder = TimeLSTMCell_2(hidden_size)

        else:
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)

        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    # DECODER
    def decode(self, input_x):
        if self.model_type == 'LSTM':
            sequence_time, h, c = input_x
            output, state = self.LSTM_decoder(sequence_time, [h, c])
        else:
            sequence_time, input_t, h, c = input_x
            output, state = self.LSTM_decoder([sequence_time, input_t], [h, c])

        x_1 = self.dense1(output)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2)
        return x_3, state[0], state[1]

    # 最后在call中整合并输出
    def call(self, encode_h, predicted_visit=1, batch=0, training=None, mask=None):
        batch = batch
        predicted_trajectory = np.zeros(shape=(batch, 0, self.feature_dims))
        predicted_trajectory_decode_h = np.zeros(shape=(batch, 0, self.hidden_size))
        decode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        decode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        if self.model_type == 'LSTM':
            context_state = encode_h
            for predicted_visit_ in range(predicted_visit):
                # h_j from 1 to j
                predicted_next_sequence, decode_h, decode_c = self.decode([context_state, decode_h, decode_c])
                context_state = tf.zeros_like(encode_h)
                predicted_trajectory_decode_h = tf.concat(
                    (predicted_trajectory_decode_h, tf.reshape(decode_h, [decode_h.shape[0], 1, decode_h.shape[1]])),
                    axis=1)
                predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, self.feature_dims])
                predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)
        else:
            context_state, input_day_train = encode_h
            for predicted_visit_ in range(predicted_visit):
                visit_interval = input_day_train[:, predicted_visit_:predicted_visit_+1]
                predicted_next_sequence, decode_h, decode_c = self.decode([context_state, visit_interval, decode_h, decode_c])
                context_state = tf.zeros_like(context_state)
                predicted_trajectory_decode_h = tf.concat(
                    (predicted_trajectory_decode_h, tf.reshape(decode_h, [decode_h.shape[0], 1, decode_h.shape[1]])),
                    axis=1)
                predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, self.feature_dims])
                predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)

        return predicted_trajectory, predicted_trajectory_decode_h

class Decoder2(tf.keras.Model):

    def __init__(self, hidden_size, feature_dims, model_type):
        super(Decoder2, self).__init__(name='Decoder')
        self.feature_dims = feature_dims
        # 首先定义各种layer层的各部分
        # ENCODER
        self.model_type = model_type
        self.hidden_size = hidden_size
        if model_type == 'LSTM':
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)
        elif model_type == 'TimeLSTM1':
            self.LSTM_decoder = TimeLSTMCell_1(hidden_size)
        elif model_type == 'TimeLSTM2':
            self.LSTM_decoder = TimeLSTMCell_2(hidden_size)
        elif model_type == 'TimeLSTM3':
            self.LSTM_decoder = TimeLSTMCell_3(hidden_size)
        else:
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)

        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

    # DECODER
    def decode(self, input_x):
        if self.model_type == 'LSTM':
            sequence_time, h, c = input_x
            output, state = self.LSTM_decoder(sequence_time, [h, c])
        else:
            sequence_time, input_t, h, c = input_x
            output, state = self.LSTM_decoder([sequence_time, input_t], [h, c])

        x_1 = self.dense1(output)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2)
        return x_3, state[0], state[1]

    # 最后在call中整合并输出
    def call(self, encode_h_list, predicted_visit=1, batch=0, training=None, mask=None):
        batch = batch
        predicted_trajectory = np.zeros(shape=(batch, 0, self.feature_dims))
        predicted_trajectory_decode_h = np.zeros(shape=(batch, 0, self.hidden_size))
        decode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        decode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        if self.model_type == 'LSTM':

            for predicted_visit_ in range(predicted_visit):
                # h_j from 1 to j
                context_state = encode_h_list[:,predicted_visit_]
                predicted_next_sequence, decode_h, decode_c = self.decode([context_state, decode_h, decode_c])
                predicted_trajectory_decode_h = tf.concat(
                    (predicted_trajectory_decode_h, tf.reshape(decode_h, [decode_h.shape[0], 1, decode_h.shape[1]])),
                    axis=1)
                predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, self.feature_dims])
                predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)
        else:
            context_state, input_day_train = encode_h_list
            for predicted_visit_ in range(predicted_visit):
                visit_interval = input_day_train[:, predicted_visit_, :]
                predicted_next_sequence, decode_h, decode_c = self.decode([context_state, visit_interval, decode_h, decode_c])
                context_state = tf.zeros_like(context_state)
                predicted_trajectory_decode_h = tf.concat(
                    (predicted_trajectory_decode_h, tf.reshape(decode_h, [decode_h.shape[0], 1, decode_h.shape[1]])),
                    axis=1)
                predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, self.feature_dims])
                predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)

        return predicted_trajectory, predicted_trajectory_decode_h

class SNMTDecoder(tf.keras.Model):

    def __init__(self, hidden_size, feature_dims, model_type):
        super(SNMTDecoder, self).__init__(name='SNMTDecoder')
        self.feature_dims = feature_dims
        # 首先定义各种layer层的各部分
        # ENCODER
        self.hidden_size = hidden_size
        if model_type == 'LSTM':
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)
        elif model_type == 'TimeLSTM1':
            self.LSTM_decoder = TimeLSTMCell_1(hidden_size)
        else:
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)

        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=1)

    # DECODER
    def decode(self, input_x):
        sequence_time, h, c = input_x
        output, state = self.LSTM_decoder(sequence_time, [h, c])

        x_1 = self.dense1(output)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2)
        return x_3, state[0], state[1]

    # 最后在call中整合并输出
    def call(self, encoder_input, predicted_visit=1, batch=0, train_flag=True, training=None, mask=None):
        batch = batch
        context_state_list = np.zeros(shape=(batch, 0, self.hidden_size))
        if train_flag:
            context_state_list, encode_h = encoder_input
        else:
            encode_h = encoder_input
        predicted_trajectory = np.zeros(shape=(batch, 0, 1))
        predicted_next_sequence = np.zeros(shape=(batch, 0, 1))
        predicted_trajectory_decode_h = np.zeros(shape=(batch, 0, self.hidden_size))
        decode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        decode_h = encode_h

        for predicted_visit_ in range(predicted_visit):
             # h_j from 1 to j
            if train_flag:
                context_state = context_state_list[:, predicted_visit_]
            else:
                if predicted_visit_ == 0:
                    context_state = np.zeros(shape=(batch, 1))
                else:
                    context_state = tf.reshape(predicted_next_sequence, [batch, 1])
            predicted_next_sequence, decode_h, decode_c = self.decode([context_state, decode_h, decode_c])
            predicted_trajectory_decode_h = tf.concat(
                (predicted_trajectory_decode_h, tf.reshape(decode_h, [decode_h.shape[0], 1, decode_h.shape[1]])),
                axis=1)
            predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, 1])
            predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)
        return predicted_trajectory, predicted_trajectory_decode_h


class USNMTDecoder(tf.keras.Model):
    def __init__(self, hidden_size, feature_dims, model_type):
        super(USNMTDecoder, self).__init__(name='USNMTDecoder')
        self.feature_dims = feature_dims
        # 首先定义各种layer层的各部分
        # ENCODER
        self.hidden_size = hidden_size
        if model_type == 'LSTM':
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)
        elif model_type == 'TimeLSTM1':
            self.LSTM_decoder = TimeLSTMCell_1(hidden_size)
        else:
            self.LSTM_decoder = tf.keras.layers.LSTMCell(hidden_size)

        self.dense1 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)

        self.dense2 = tf.keras.layers.Dense(units=feature_dims, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=feature_dims)

    # DECODER
    def decode(self, input_x):
        sequence_time, h, c = input_x
        output, state = self.LSTM_decoder(sequence_time, [h, c])

        x_1 = self.dense1(output)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2)
        return x_3, state[0], state[1]

    # 最后在call中整合并输出
    def call(self, encoder_input, predicted_visit=1, batch=0, training=None, mask=None):
        batch = batch
        context_state_list, mask_start, mask_end, encode_h = encoder_input
        predicted_trajectory = np.zeros(shape=(batch, 0, 1))
        predicted_next_sequence = np.zeros(shape=(batch, 0, 1))
        predicted_trajectory_decode_h = np.zeros(shape=(batch, 0, self.hidden_size))
        decode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        decode_h = encode_h

        for predicted_visit_ in range(predicted_visit):
            if predicted_visit_ >= mask_start and predicted_visit_<mask_end:
                context_state = context_state_list[:, predicted_visit_-1,:]
            else:
                context_state = tf.zeros_like(context_state_list[:, predicted_visit_,:])


            predicted_next_sequence, decode_h, decode_c = self.decode([context_state, decode_h, decode_c])
            predicted_trajectory_decode_h = tf.concat(
                (predicted_trajectory_decode_h, tf.reshape(decode_h, [decode_h.shape[0], 1, decode_h.shape[1]])),
                axis=1)
            predicted_next_sequence = tf.reshape(predicted_next_sequence, [batch, -1, 1])
            predicted_trajectory = tf.concat((predicted_trajectory, predicted_next_sequence), axis=1)
        return predicted_trajectory, predicted_trajectory_decode_h