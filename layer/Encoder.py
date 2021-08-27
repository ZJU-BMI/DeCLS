import pandas as pd
import numpy as np
from tensorflow.keras import *
from tensorflow import keras
from layer.TimeLSTMCell_2 import *


# 对于
# input: seq_x train_flag size prev_len pred_len
# output: gen_x pre_y list_h list_c


# 第一部分S2S
class Encoder(tf.keras.Model):

    def __init__(self, hidden_size, model_type):
        super(Encoder, self).__init__(name='Encoder')

        # 首先定义各种layer层的各部分
        # ENCODER
        self.hidden_size = hidden_size
        self.model_type = model_type
        if model_type == 'LSTM':
            self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size, recurrent_activation='tanh', implementation=1)
     
        elif model_type == 'TimeLSTM2':
            self.LSTM_Cell_encode = TimeLSTMCell_2(hidden_size)
       
        else:
            self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

    # ENCODER
    def encode(self, input_x):
        if self.model_type == 'LSTM':
            sequence_time, h, c = input_x
            output, state = self.LSTM_Cell_encode(sequence_time, [h, c])
        else:
            sequence_time, input_t, h, c = input_x
            output, state = self.LSTM_Cell_encode([sequence_time, input_t], [h, c])
        return state[0], state[1]

    # 最后在call中整合并输出
    def call(self, input_x_trains, batch=0, train_flag=True, training=None, mask=None):
        if self.model_type == 'LSTM':
            input_x_train = input_x_trains
            # h_shape = self.hidden_size
            h_shape = self.hidden_size+input_x_trains.shape[2]
            trajectory_encode_h_list = np.zeros(shape=(batch, 0, h_shape))
            trajectory_encode_c_list = np.zeros(shape=(batch, 0, self.hidden_size))
            trajectory_len = input_x_train.shape[1]
            encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            for visit in range(trajectory_len):
                sequence_time = input_x_train[:, visit, :]  # y_j
                encode_h, encode_c = self.encode([sequence_time, encode_h, encode_c])
                # total_h = tf.concat([sequence_time, encode_h], axis=1)
                total_h = tf.concat([sequence_time, encode_h], axis=1)
                # trajectory_encode_h_list = tf.concat(
                #     (trajectory_encode_h_list, tf.reshape(total_h, [encode_h.shape[0], 1,-1])),
                #     axis=1)
                trajectory_encode_h_list = tf.concat(
                    (trajectory_encode_h_list, tf.reshape(total_h, [encode_h.shape[0], 1, -1])),
                    axis=1)
                trajectory_encode_c_list = tf.concat(
                    (trajectory_encode_c_list, tf.reshape(encode_c, [encode_c.shape[0], 1, encode_c.shape[1]])),
                    axis=1)
            context_state = tf.concat([input_x_train[:, -1, :], encode_h], axis=1)  # h_j from 1 to j
        else:
            input_x_train, input_day_train = input_x_trains
            trajectory_encode_h_list = np.zeros(shape=(batch, 0, self.hidden_size))
            trajectory_encode_c_list = np.zeros(shape=(batch, 0, self.hidden_size))
            trajectory_len = input_x_train.shape[1]
            encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
            for visit in range(trajectory_len):
                sequence_time = input_x_train[:, visit, :]  # y_j
                visit_interval = np.reshape(input_day_train[:, visit], (-1,1))
                encode_h, encode_c = self.encode([sequence_time, visit_interval, encode_h, encode_c])
                trajectory_encode_h_list = tf.concat(
                    (trajectory_encode_h_list, tf.reshape(encode_h, [encode_h.shape[0], 1, encode_h.shape[1]])),
                    axis=1)
                trajectory_encode_c_list = tf.concat(
                    (trajectory_encode_c_list, tf.reshape(encode_c, [encode_c.shape[0], 1, encode_c.shape[1]])),
                    axis=1)
            context_state = encode_h  # h_j from 1 to j

        return context_state, trajectory_encode_h_list
