import matplotlib.pyplot as plt
import datetime

import os
import tensorflow as tf
import data_reader as reader
from model import Model

PATH_TRAIN = "data/train"
PATH_TEST = "data/test"
PATH_SUMMARY = "log/summary"
PATH_CHECKPOINT = "log/data"

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
CAPACITY = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 10
N_THREADS = 10
EPOCH = 10
TEST_SIZE = 1000


def next_batch(images, labels, batch_size=BATCH_SIZE):
    return reader.get_batch(
        images, labels, IMAGE_WIDTH, IMAGE_HEIGHT, batch_size, CAPACITY, N_THREADS)


if __name__ == '__main__':
    images, labels = reader.read_data(PATH_TRAIN)
    train_images, train_labels = images[:-TEST_SIZE], labels[:-TEST_SIZE]
    batch_train = next_batch(train_images, train_labels)
    test_images, test_labels = images[-TEST_SIZE:], labels[-TEST_SIZE:]
    batch_test = next_batch(test_images, test_labels)
    model = Model(IMAGE_WIDTH, IMAGE_HEIGHT, 3, 2, LEARNING_RATE)
    train_step = model.train_step()

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(PATH_CHECKPOINT)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print "Load train data."
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    summary_loss = tf.summary.scalar("loss", model.loss)
    summary_accuracy = tf.summary.scalar("accuracy", model.accuracy())
    summary_merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(PATH_SUMMARY, graph=tf.get_default_graph())

    test_batch_images, test_batch_labels = session.run(batch_test)
    try:
        step = 1
        for epoch in range(1, EPOCH + 1):
            for i in xrange(len(train_labels) / BATCH_SIZE):
                batch_images, batch_labels = session.run(batch_train)
                train_step.run(feed_dict={
                    model.input: batch_images,
                    model.output: batch_labels,
                    model.keep_prob: 1
                })

                if step % 1 == 0:
                    loss = model.loss.eval(feed_dict={
                        model.input: test_batch_images,
                        model.output: test_batch_labels,
                        model.keep_prob: 1
                    })
                    print "Time: %s, Epoch %d, Step %d, Loss %f" % \
                          (datetime.datetime.now(), epoch, step, loss)
                    # print "Time: %s, Epoch %d, Step %d" % (datetime.datetime.now(), epoch, step)
                if step % 10 == 0:
                    summary_str = session.run(summary_merged, feed_dict={
                        model.input: test_batch_images,
                        model.output: test_batch_labels,
                        model.keep_prob: 1
                    })
                    summary_writer.add_summary(summary_str, step)
                if step % 100 == 0:
                    saver.save(session, os.path.join(PATH_CHECKPOINT, "model"), step)
                step += 1
    except tf.errors.OutOfRangeError:
        print "Error."
    finally:
        coord.should_stop()
    coord.join()
