import tensorflow as tf


class Network:
    def __init__(self, input):
        self.__data = input

    def __weight_variables(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def __biase_variables(self, shape):
        return tf.Variable(tf.constant(0.01, shape=shape))

    def fc(self, input_size, out_size, activation=tf.nn.relu, name=None):
        data_shape = self.__data.get_shape()
        if len(data_shape) == 4:
            self.__data = tf.reshape(self.__data, [-1, input_size])
        W = self.__weight_variables([input_size, out_size])
        b = self.__biase_variables([out_size])
        self.__data = tf.matmul(self.__data, W) + b
        if activation is not None:
            self.__data = activation(self.__data)

    def conv2d(self, filter_shape, strides, relu=True, name=None):
        W = self.__weight_variables(filter_shape)
        b = self.__biase_variables([filter_shape[-1]])
        self.__data = tf.nn.conv2d(self.__data, W, strides=strides, padding="SAME", name=name) + b
        if relu:
            self.relu()

    def max_pool(self, ksize, strides, name=None):
        self.__data = tf.nn.max_pool(self.__data, ksize, strides, padding="SAME", name=name)

    def dropout(self, keep_prob, name=None):
        self.__data = tf.nn.dropout(self.__data, keep_prob, name=name)

    def softmax(self, name=None):
        data_shape = self.__data.get_shape()
        if len(data_shape) == 4:
            self.__data = tf.reshape(self.__data, [-1, data_shape[1] * data_shape[2] * data_shape[3]])
        self.__data = tf.nn.softmax(self.__data, name=name)

    def relu(self, name=None):
        self.__data = tf.nn.relu(self.__data, name)

    def output(self):
        return self.__data
