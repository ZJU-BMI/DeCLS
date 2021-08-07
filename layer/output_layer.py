
import tensorflow as tf
from layer.TimeLSTMCell_2 import *
from layer.TimeLSTMCell_3 import *
# class ENCODER(tf.keras.Model):
#     def __init__(self, hidden_size):
#         super(ENCODER, self).__init__(name='ENCODER')
#         # 首先定义各种layer层的各部分
#         # ENCODER
#         self.hidden_size = hidden_size
#         self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size, implementation=2)
#
#     # ENCODER
#     def encode(self, input_x):
#         sequence_time, h, c = input_x
#         output, state = self.LSTM_Cell_encode(sequence_time, [h, c])
#         return state[0], state[1]
#
#     # 最后在call中整合并输出
#     def call(self, input_x_train, batch=0, training=None, mask=None):
#         visit = input_x_train.shape[1]
#         encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
#         encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
#         for v in range(visit):
#             sequence_time = input_x_train[:, v, :]
#             encode_h, encode_c = self.encode([sequence_time, encode_h, encode_c])
#
#
#         context_state = encode_h  # h_j from 1 to j
#         return context_state


class SAP2(tf.keras.Model):

    def __init__(self, hidden_size, model_type='LSTM'):
        super(SAP2, self).__init__(name='SAP')
        self.hidden_size = hidden_size
        self.model_type = model_type
        if model_type == 'LSTM':
            self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)
        elif model_type == 'TimeLSTM1':
            self.LSTM_Cell_encode = TimeLSTMCell_1(hidden_size)
        elif model_type == 'TimeLSTM2':
            self.LSTM_Cell_encode = TimeLSTMCell_2(hidden_size)
        elif model_type == 'TimeLSTM3':
            self.LSTM_Cell_encode = TimeLSTMCell_3(hidden_size)
        else:
            self.LSTM_Cell_encode = tf.keras.layers.LSTMCell(hidden_size)

        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.tanh)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(units=1)

    # ENCODER
    def encode(self, input_all):
        if self.model_type == 'LSTM':
            sequence_visit, h, c = input_all
            output, state = self.LSTM_Cell_encode(sequence_visit, [h, c])
        else:
            sequence_visit, input_t, h, c = input_all
            output, state = self.LSTM_Cell_encode([sequence_visit, input_t], [h, c])
        return state[0], state[1]

    def mlp(self, input_h):
        x_1 = self.dense1(input_h)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2)
        return x_3

    def call(self, input_x_trains, batch=0, train_flag=True, training=None, mask=None):
        trajectory_encode_h_list = np.zeros(shape=(batch, 0, self.hidden_size))
        trajectory_y_list =  np.zeros(shape=(batch, 0, 1))
        trajectory_encode_c_list = np.zeros(shape=(batch, 0, self.hidden_size))
        encode_c = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))
        encode_h = tf.Variable(tf.zeros(shape=[batch, self.hidden_size]))

        if self.model_type == 'LSTM':
            input_x_train = input_x_trains

            trajectory_len = input_x_train.shape[1]
            for visit in range(trajectory_len):
                sequence_time = input_x_train[:, visit, :]  # y_j

                encode_h, encode_c = self.encode([sequence_time, encode_h, encode_c])
                predict_survival = self.mlp(encode_h)
                trajectory_y_list = tf.concat(
                    (trajectory_y_list, tf.reshape(predict_survival, [predict_survival.shape[0], 1, predict_survival.shape[1]])),
                    axis=1)
                trajectory_encode_h_list = tf.concat(
                    (trajectory_encode_h_list, tf.reshape(encode_h, [encode_h.shape[0], 1, encode_h.shape[1]])),
                    axis=1)
                trajectory_encode_c_list = tf.concat(
                    (trajectory_encode_c_list, tf.reshape(encode_c, [encode_c.shape[0], 1, encode_c.shape[1]])),
                    axis=1)
            context_state = encode_h  # h_j from 1 to j
        else:
            input_x_train, input_day_train = input_x_trains
            trajectory_len = input_x_train.shape[1]
            for visit in range(trajectory_len):
                sequence_time = input_x_train[:, visit, :]  # y_j
                visit_interval = input_day_train[:, visit,:]
                encode_h, encode_c = self.encode([sequence_time, visit_interval, encode_h, encode_c])
                predict_survival = self.mlp(encode_h)
                trajectory_y_list = tf.concat(
                    (trajectory_y_list,
                     tf.reshape(predict_survival, [predict_survival.shape[0], 1, predict_survival.shape[1]])),
                    axis=1)
                trajectory_encode_h_list = tf.concat(
                    (trajectory_encode_h_list, tf.reshape(encode_h, [encode_h.shape[0], 1, encode_h.shape[1]])),
                    axis=1)
                trajectory_encode_c_list = tf.concat(
                    (trajectory_encode_c_list, tf.reshape(encode_c, [encode_c.shape[0], 1, encode_c.shape[1]])),
                    axis=1)
            context_state = encode_h  # h_j from 1 to j

        return trajectory_y_list

# 生存分析 survival analysis predict
class SAP(tf.keras.Model):
    def __init__(self, hidden_size):
        super(SAP, self).__init__(name='SAP')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense3 = tf.keras.layers.Dense(units=1)

    def call(self, input_x_train, training=None, mask=None):
        x_1 = self.dense1(input_x_train)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2)

        # x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train))
        # x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1))
        # x_3 = self.dense3(x_2)
        return x_3

# class FC_SAP(tf.keras.Model):
#
#     def __init__(self, hidden_size):
#         super(FC_SAP, self).__init__(name='FC_SAP')
#         self.hidden_size = hidden_size
#         self.dense1 = tf.keras.layers.Dense(units=hidden_size)
#         self.dense2 = tf.keras.layers.Dense(units=1)
#
#     def call(self, input_x_train):
#         # x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train),training=training)
#         x_2 = self.dense2(input_x_train)
#         return input_x_train

class FC_SAP2(tf.keras.Model):

    def __init__(self, hidden_size, num_category):
        super(FC_SAP2, self).__init__(name='FC_SAP2')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense3 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)

        self.dense4 = tf.keras.layers.Dense(units=num_category, activation=tf.nn.softmax)

    def call(self, input_x_train):
        # x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train),training=training)
        # x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1),training=training)
        # x_3 = tf.keras.layers.BatchNormalization()(self.dense3(x_2),training=training)
        x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train))
        x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1))
        x_3 = tf.keras.layers.BatchNormalization()(self.dense3(x_2))
        return self.dense4(x_3)
class FC_SAP(tf.keras.Model):

    def __init__(self, hidden_size, num_category):
        super(FC_SAP, self).__init__(name='FC_SAP')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense3 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)

        self.dense4 = tf.keras.layers.Dense(units=num_category, activation=tf.nn.softmax)

    def call(self, input_x_train):
        # x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train),training=training)
        # x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1),training=training)
        # x_3 = tf.keras.layers.BatchNormalization()(self.dense3(x_2),training=training)
        x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train))
        x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1))
        x_3 = tf.keras.layers.BatchNormalization()(self.dense3(x_2))
        return x_3

class MLP(tf.keras.Model):
    def __init__(self, hidden_size):
        super(MLP, self).__init__(name='MLP')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)

        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense3 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)

        self.dense4 = tf.keras.layers.Dense(units=1)

    def call(self, input_x_train):
        # x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train),training=training)
        # x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1),training=training)
        # x_3 = tf.keras.layers.BatchNormalization()(self.dense3(x_2),training=training)
        x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train))
        x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1))
        x_3 = tf.keras.layers.BatchNormalization()(self.dense3(x_2))
        return self.dense4(x_3)




class DEEPSURV(tf.keras.Model):
    def __init__(self, hidden_size):
        super(DEEPSURV, self).__init__(name='DEEPSURV')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.elu)
        self.dense3 = tf.keras.layers.Dense(units=1)

    def call(self, input_x_train, training=None, mask=None):


        x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train))
        x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1))
        x_3 = self.dense3(x_2)
        # x_1 = self.dense1(input_x_train)
        # x_2 = self.dense2(x_1)
        # x_3 = self.dense3(x_2)
        return x_3


class MLP2(tf.keras.Model):
    def __init__(self, hidden_size):
        super(MLP2, self).__init__(name='MLP2')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.softmax)

    def call(self, input_x_train, training=None, mask=None):
        x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train))
        return x_1

class LR(tf.keras.Model):
    def __init__(self, hidden_size):
        super(LR, self).__init__(name='LR')
        self.hidden_size = hidden_size
        self.dense = tf.keras.layers.Dense(units=1)

    def call(self, input_x_train, training=None, mask=None):
        x_1 = self.dense(input_x_train)
        return x_1


# 鉴别器
class DISCRIMINATOR(tf.keras.Model):
    def __init__(self, hidden_size):
        super(DISCRIMINATOR, self).__init__(name='DISCRIMINATOR')
        self.hidden_size = hidden_size
        self.dense1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

    def call(self, input_x_train, training=None, mask=None):
        x_1 = self.dense1(input_x_train)
        x_2 = self.dense2(x_1)
        x_3 = self.dense3(x_2)
        return x_3