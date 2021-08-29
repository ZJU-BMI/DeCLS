
import tensorflow as tf
from layer.TimeLSTMCell_2 import *


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
        x_1 = tf.keras.layers.BatchNormalization()(self.dense1(input_x_train))
        x_2 = tf.keras.layers.BatchNormalization()(self.dense2(x_1))
        x_3 = tf.keras.layers.BatchNormalization()(self.dense3(x_2))
        return self.dense4(x_3)


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