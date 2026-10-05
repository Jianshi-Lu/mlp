# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import pickle
import gzip
import numpy as np
import os
from mlp import DEFAULT_SEED


_DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'data'))


def _data_dir():
    """Return the configured data directory or this checkout's data folder."""
    return os.environ.get('MLP_DATA_DIR', _DEFAULT_DATA_DIR)


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        assert max_num_batches != 0 and not max_num_batches < -1, (
            'max_num_batches should be -1 or > 0')
        self.max_num_batches = max_num_batches
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.reset()

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def reset(self):
        """Resets the provider to the initial state to use in a new epoch."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def shuffle(self):
        """Randomly shuffles order of data."""
        new_order = self.rng.permutation(self.inputs.shape[0])
        self.inputs = self.inputs[new_order]
        self.targets = self.targets[new_order]

    # 应用老师的 Python 3 迭代协议：所有子类都能使用 next(provider) 和 for 循环。
    # self.next() 会调用当前实例的实现，因此 MNIST 仍会执行自己的 one-hot 转换。
    def __next__(self):
        return self.next()

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so reset
            # the dataset for another pass and indicate iteration is at end
            self.reset()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch


class MNISTDataProvider(DataProvider):
    """Data provider for MNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'eval'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 10
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        data_path = os.path.join(
            _data_dir(), 'mnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(MNISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MNISTDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def __next__(self):
        return self.next()

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        # 保留我们的 Lab 1 实现：单位矩阵的第 k 行就是类别 k 的 one-hot 向量。
        # 已核对所有 0～9 类别；输出形状为 (num_data, 10)，类型保持 float32。
        identity_matrix = np.eye(
            self.num_classes,
            dtype=np.float32
        )
        one_hot_targets = identity_matrix[int_targets]
        return one_hot_targets

        # 老师的 Lab 2 参考实现（注释保存，不重复运行）：
        # 先建全零矩阵，再把每行对应类别的位置设为 1；数值与我们的实现一致。
        # np.zeros 未指定 dtype 时默认是 float64，我们继续保留 float32。
        # one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        # one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        # return one_of_k_targets

class MetOfficeDataProvider(DataProvider):
    """South Scotland Met Office weather data provider."""

    def __init__(self, window_size, batch_size=10, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new Met Office data provider object.

        Args:
            window_size (int): Size of windows to split weather time series
               data into. The constructed input features will be the first
               `window_size - 1` entries in each window and the target outputs
               the last entry in each window.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # 恢复初始化：合并前的 Lab 1 和老师版本都有这两行，手动解决冲突时遗漏了。
        # 后面的窗口循环读取 self.window_size；只传入参数而不保存，会触发 AttributeError。
        self.window_size = window_size
        assert window_size > 1, 'window_size must be at least 2.'
        data_path = os.path.join(_data_dir(), 'HadSSP_daily_qc.txt')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        # 1. 保留我们的读取方法：文件前三行是说明，之后每行是 年、月、1～31 日降雨量。
        # 去掉年和月两列，保留全部 31 个日期列；-99.99 是本文件实际使用的缺失值。
        raw_data = np.loadtxt(data_path, skiprows=3)
        daily_values = raw_data[:, 2:].reshape(-1)
        daily_values = daily_values[daily_values != -99.99]

        # 2. 用整条有效序列的均值和标准差，把每一天的值换算成“偏离均值多少个标准差”。
        # 与老师分开计算 mean 和 std 的公式一致；不是按月份或按年份单独计算。
        daily_values = (daily_values - daily_values.mean()) / daily_values.std()

        # 3. 保留我们的列表切片方法：相邻窗口起点只移动 1 个位置。
        # 长度为 N、窗口为 W 时共有 N - W + 1 个窗口；已逐值核对窗口内容和先后顺序。
        window_list = []
        for i in range(len(daily_values) - self.window_size + 1):
            window_list.append(daily_values[i:i + self.window_size])
        windows = np.array(window_list)

        # 4. 每行是一个窗口：前 W - 1 个值作为输入，最后一个值作为预测目标。
        inputs = windows[:, :-1]
        targets = windows[:, -1]

        # 数据预处理完成后，显式调用父类初始化，保存输入/目标并设置分批、洗牌和迭代状态。
        # 父类不会重新读取天气文件；传给父类的是这里已经构造好的两个数组。
        super(MetOfficeDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng
        )

        # 老师的 Lab 2 原始参考实现（整段注释保存，不重复运行）：
        # 注意第一行 usecols=range(2, 32) 只取索引 2～31，会漏掉索引 32 的第 31 天。
        # 本文件共 33 列，其中第 31 天有 593 条有效记录，所以保留我们的完整读取方法。
        # 若采用 usecols，完整日期列应写 range(2, 33)。
        # 在保留相同日期序列的前提下，老师的过滤、标准化和 as_strided 窗口数值与我们一致。
        # as_strided 创建共享内存的窗口视图；我们的 np.array(window_list) 生成独立数组。
        # raw = np.loadtxt(data_path, skiprows=3, usecols=range(2, 32))
        # assert window_size > 1, 'window_size must be at least 2.'
        # self.window_size = window_size
        # # filter out all missing datapoints and flatten to a vector
        # filtered = raw[raw >= 0].flatten()
        # # normalise data to zero mean, unit standard deviation
        # mean = np.mean(filtered)
        # std = np.std(filtered)
        # normalised = (filtered - mean) / std
        # # create a view on to array corresponding to a rolling window
        # shape = (normalised.shape[-1] - self.window_size + 1, self.window_size)
        # strides = normalised.strides + (normalised.strides[-1],)
        # windowed = np.lib.stride_tricks.as_strided(
        #     normalised, shape=shape, strides=strides)
        # # inputs are first (window_size - 1) entries in windows
        # inputs = windowed[:, :-1]
        # # targets are last entry in windows
        # targets = windowed[:, -1]
        # super(MetOfficeDataProvider, self).__init__(
        #     inputs, targets, batch_size, max_num_batches, shuffle_order, rng)
    def __next__(self):
        return self.next()


class CCPPDataProvider(DataProvider):

    def __init__(self, which_set='train', input_dims=None, batch_size=10,
                 max_num_batches=-1, shuffle_order=True, rng=None):
        """Create a new Combined Cycle Power Plant data provider object.

        Args:
            which_set: One of 'train' or 'valid'. Determines which portion of
                data this object should provide.
            input_dims: Which of the four input dimension to use. If `None` all
                are used. If an iterable of integers are provided (consisting
                of a subset of {0, 1, 2, 3}) then only the corresponding
                input dimensions are included.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        data_path = os.path.join(_data_dir(), 'ccpp_data.npz')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # check a valid which_set was provided
        assert which_set in ['train', 'valid'], (
            'Expected which_set to be either train or valid '
            'Got {0}'.format(which_set)
        )
        # 应用老师新增的 CCPP 数据提供器，但修正原版的列选择检查。
        # 老师原代码（注释保留）：
        # if not input_dims is not None:
        #     input_dims = set(input_dims)
        #     assert input_dims.issubset({0, 1, 2, 3}), (
        #         'input_dims should be a subset of {0, 1, 2, 3}'
        #     )
        # 问题 1：原条件等同于 input_dims is None，默认值会进入 set(None)，触发 TypeError。
        # 问题 2：即使改对条件，set 也不能直接作为 NumPy 列索引，而且会丢失选择顺序。
        # 正确做法：None 保留全部四列；显式选择时用列表索引，只临时用集合验证列号。
        if input_dims is not None:
            input_dims = list(input_dims)
            assert set(input_dims).issubset({0, 1, 2, 3}), (
                'input_dims should be a subset of {0, 1, 2, 3}'
            )
        loaded = np.load(data_path)
        inputs = loaded[which_set + '_inputs']
        if input_dims is not None:
            inputs = inputs[:, input_dims]
        targets = loaded[which_set + '_targets']
        super(CCPPDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)
