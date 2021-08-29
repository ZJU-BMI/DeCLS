import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
from collections import defaultdict
from sklearn.preprocessing import normalize
#


class DataSet(object):
    """
    输入特征与标签作为一个集合，可以用于小批次训练

    """

    def __init__(self, x, t, y):
        self._x = x
        self._y = y
        self._t = t
        self._num_example = self._x.shape[0]
        self._index = np.arange(self._num_example)
        self._epoch_completed = 0
        self._index_in_epoch = 0
        if self._x.shape[0] != self._y.shape[0]:
            raise ValueError('The num example of x is not equal to y ')

    def next_batch(self, batch_size):
        # batch_size 输入标准未考虑
        if batch_size < 0 or batch_size > self._num_example:
            raise ValueError('The size of one batch: {} should be less than the total number of '
                             'data: {}'.format(batch_size, self.num_examples))
            # batch_size = self._num_example
            # self._shuffle()
        start = self._index_in_epoch
        if start + batch_size > self._num_example:
            self._index_in_epoch = self._num_example
            x_batch_rest = self._x[start:self._index_in_epoch]
            y_batch_rest = self._y[start:self._index_in_epoch]
            t_batch_rest = self._t[start:self._index_in_epoch]
            self._epoch_completed += 1
            self._index_in_epoch = 0
            self._shuffle()
            rest = start + batch_size - self._num_example
            x_batch_new = self._x[self._index_in_epoch:self._index_in_epoch + rest]
            y_batch_new = self._y[self._index_in_epoch:self._index_in_epoch + rest]
            t_batch_new = self._t[self._index_in_epoch:self._index_in_epoch + rest]
            self._index_in_epoch += rest

            return np.concatenate((x_batch_rest, x_batch_new), axis=0), \
                   np.concatenate((t_batch_rest, t_batch_new), axis=0), \
                   np.concatenate((y_batch_rest, y_batch_new), axis=0)
        else:
            self._index_in_epoch = start + batch_size
            x_batch = self._x[start:self._index_in_epoch]
            t_batch = self._t[start:self._index_in_epoch]
            y_batch = self._y[start:self._index_in_epoch]
            return x_batch, t_batch, y_batch

    def _shuffle(self):
        index = np.arange(self._num_example)
        np.random.shuffle(index)
        self._index = index
        self._x = self._x[index]
        self._y = self._y[index]
        self._t = self._t[index]

    @property
    def num_examples(self):
        return self._num_example

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def t(self):
        return self._t

    @property
    def index(self):
        return self._index

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


class DataSetWithMask2(object):
    """

    输入特征与标签作为一个集合，可以用于小批次训练
    相比于DataSet， 新增了一个属性mask
    mask代表数据是否是填零插补
    额外增加了visit的时间信息
    """

    def __init__(self, x, t, y, day, mask1, mask2, mask3):
        self._x = x
        self._y = y
        self._t = t
        self._day = day
        self._mask1 = mask1
        self._mask2 = mask2
        self._mask3 = mask3
        self._num_example = self._x.shape[0]
        self._index = np.arange(self._num_example)
        self._epoch_completed = 0
        self._index_in_epoch = 0
        self._shuffle()
        if self._x.shape[0] != self._y.shape[0]:
            raise ValueError('The num example of x is not equal to y ')

    def next_batch(self, batch_size):
        # batch_size 输入标准未考虑
        if batch_size < 0 or batch_size > self._num_example:
            raise ValueError('The size of one batch: {} should be less than the total number of '
                             'data: {}'.format(batch_size, self.num_examples))
            # batch_size = self._num_example
            # self._shuffle()
        start = self._index_in_epoch
        if start + batch_size > self._num_example:
            self._index_in_epoch = self._num_example
            x_batch_rest = self._x[start:self._index_in_epoch]
            y_batch_rest = self._y[start:self._index_in_epoch]
            t_batch_rest = self._t[start:self._index_in_epoch]
            mask_batch_rest1 = self._mask1[start:self._index_in_epoch]
            mask_batch_rest2 = self._mask2[start:self._index_in_epoch]
            mask_batch_rest3 = self._mask3[start:self._index_in_epoch]
            day_batch_rest = self._day[start:self._index_in_epoch]
            self._epoch_completed += 1
            self._index_in_epoch = 0
            self._shuffle()
            rest = start + batch_size - self._num_example
            x_batch_new = self._x[self._index_in_epoch:self._index_in_epoch + rest]
            y_batch_new = self._y[self._index_in_epoch:self._index_in_epoch + rest]
            t_batch_new = self._t[self._index_in_epoch:self._index_in_epoch + rest]
            day_batch_new = self._day[self._index_in_epoch:self._index_in_epoch + rest]
            mask_batch_new1 = self._mask1[self._index_in_epoch:self._index_in_epoch + rest]
            mask_batch_new2 = self._mask2[self._index_in_epoch:self._index_in_epoch + rest]
            mask_batch_new3 = self._mask3[self._index_in_epoch:self._index_in_epoch + rest]
            self._index_in_epoch += rest

            return np.concatenate((x_batch_rest, x_batch_new), axis=0), \
                   np.concatenate((t_batch_rest, t_batch_new), axis=0), \
                   np.concatenate((y_batch_rest, y_batch_new), axis=0), \
                   np.concatenate((day_batch_rest, day_batch_new), axis=0), \
                   np.concatenate((mask_batch_rest1, mask_batch_new1), axis=0), \
                   np.concatenate((mask_batch_rest2, mask_batch_new2), axis=0), \
                   np.concatenate((mask_batch_rest3, mask_batch_new3), axis=0)
        else:
            self._index_in_epoch = start + batch_size
            x_batch = self._x[start:self._index_in_epoch]
            t_batch = self._t[start:self._index_in_epoch]
            y_batch = self._y[start:self._index_in_epoch]
            day_batch = self._day[start:self._index_in_epoch]
            mask_batch1 = self._mask1[start:self._index_in_epoch]
            mask_batch2 = self._mask2[start:self._index_in_epoch]
            mask_batch3 = self._mask3[start:self._index_in_epoch]


            return x_batch, t_batch, y_batch,day_batch, mask_batch1,mask_batch2,mask_batch3

    def _shuffle(self):
        index = np.arange(self._num_example)
        np.random.shuffle(index)
        self._index = index
        self._x = self._x[index]
        self._y = self._y[index]
        self._t = self._t[index]
        self._mask1 = self._mask1[index]
        self._mask2 = self._mask2[index]
        self._mask3 = self._mask3[index]
        self._day = self._day[index]

    @property
    def num_examples(self):
        return self._num_example

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def t(self):
        return self._t

    @property
    def day(self):
        return self._day

    @property
    def mask1(self):
        return self._mask1

    @property
    def mask2(self):
        return self._mask2

    @property
    def mask3(self):
        return self._mask3

    @property
    def index(self):
        return self._index

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value

class DataSet2(object):
    """
    输入特征与标签作为一个集合，可以用于小批次训练
    相比于DataSet， 新增了一个属性day，用于输入TimeLstmCell中，作为一个门控
    day代表本次visit相对于上一次visit的时间间隔
    """

    def __init__(self, x, t, y, day):
        self._x = x
        self._y = y
        self._t = t
        self._day = day
        self._num_example = self._x.shape[0]
        self._index = np.arange(self._num_example)
        self._epoch_completed = 0
        self._index_in_epoch = 0
        if self._x.shape[0] != self._y.shape[0]:
            raise ValueError('The num example of x is not equal to y ')

    def next_batch(self, batch_size):
        # batch_size 输入标准未考虑
        if batch_size < 0 or batch_size > self._num_example:
            raise ValueError('The size of one batch: {} should be less than the total number of '
                             'data: {}'.format(batch_size, self.num_examples))
            # batch_size = self._num_example
            # self._shuffle()
        start = self._index_in_epoch
        if start + batch_size > self._num_example:
            self._index_in_epoch = self._num_example
            x_batch_rest = self._x[start:self._index_in_epoch]
            y_batch_rest = self._y[start:self._index_in_epoch]
            t_batch_rest = self._t[start:self._index_in_epoch]
            day_batch_rest = self._day[start:self._index_in_epoch]
            self._epoch_completed += 1
            self._index_in_epoch = 0
            self._shuffle()
            rest = start + batch_size - self._num_example
            x_batch_new = self._x[self._index_in_epoch:self._index_in_epoch + rest]
            y_batch_new = self._y[self._index_in_epoch:self._index_in_epoch + rest]
            t_batch_new = self._t[self._index_in_epoch:self._index_in_epoch + rest]
            day_batch_new = self._day[self._index_in_epoch:self._index_in_epoch + rest]
            self._index_in_epoch += rest

            return np.concatenate((x_batch_rest, x_batch_new), axis=0), \
                   np.concatenate((t_batch_rest, t_batch_new), axis=0), \
                   np.concatenate((y_batch_rest, y_batch_new), axis=0), \
                   np.concatenate((day_batch_rest, day_batch_new), axis=0)

        else:
            self._index_in_epoch = start + batch_size
            x_batch = self._x[start:self._index_in_epoch]
            t_batch = self._t[start:self._index_in_epoch]
            y_batch = self._y[start:self._index_in_epoch]
            day_batch = self._day[start:self._index_in_epoch]
            return x_batch, t_batch, y_batch, day_batch

    def _shuffle(self):
        index = np.arange(self._num_example)
        np.random.shuffle(index)
        self._index = index
        self._x = self._x[index]
        self._y = self._y[index]
        self._t = self._t[index]
        self._day = self._day[index]

    @property
    def num_examples(self):
        return self._num_example

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def t(self):
        return self._t

    @property
    def day(self):
        return self._day

    @property
    def index(self):
        return self._index

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value

