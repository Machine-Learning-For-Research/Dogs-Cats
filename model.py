import tensorflow as tf

import network


class Model:
    def __init__(self, image_W, image_H, image_D, n_classes, learning_rate):
        self.__image_W = image_W
        self.__image_H = image_H
        self.__image_D = image_D
        self.__learning_rate = learning_rate
        self.input = tf.placeholder(tf.float32, [None, image_H, image_W, image_D])
        self.keep_prob = tf.placeholder(tf.float32)
        self.output = tf.placeholder(tf.float32, [None, n_classes])
        self.output_predict = self.__init_network()
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.output, logits=self.output_predict
        # ))
        self.loss = -tf.reduce_sum(self.output * tf.log(self.output_predict))
        # self.loss = tf.reduce_mean(tf.square(self.output - self.output_predict))

    def __init_network(self):
        nw = network.Network(self.input)

        nw.conv2d(filter_shape=[3, 3, 3, 16], strides=[1, 1, 1, 1], name="Conv1")
        nw.max_pool(ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="Pool1")

        nw.conv2d(filter_shape=[3, 3, 16, 16], strides=[1, 1, 1, 1], name="Conv2")
        nw.max_pool(ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="Pool2")

        nw.fc(50 * 50 * 16, 256, name="Fc1")
        nw.dropout(self.keep_prob, name="Drop1")

        nw.fc(256, 2, activation=tf.nn.softmax, name="Fc3")
        return nw.output()

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.output_predict, 1), tf.argmax(self.output, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train_step(self):
        optimizer = tf.train.AdamOptimizer(self.__learning_rate)
        return optimizer.minimize(self.loss)
