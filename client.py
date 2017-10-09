import data_reader as reader
import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 3

if __name__ == '__main__':
    images, labels = reader.read_data(reader.PATH_TRAIN)
    batch_image, batch_label = reader.get_batch(images, labels, 200, 200, BATCH_SIZE, 100)

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        try:
            while not coord.should_stop() and i < 1:
                _images, _labels = session.run([batch_image, batch_label])
                for j in range(BATCH_SIZE):
                    print "Label is %d." % _labels[j]
                    # plt.imshow(_images[j])
                    # plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print "Error."
        finally:
            coord.request_stop()
        coord.join()

