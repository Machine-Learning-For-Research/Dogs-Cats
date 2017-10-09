import tensorflow as tf


if __name__ == '__main__':
    a = range(10)
    b = tf.constant(a)
    sess = tf.InteractiveSession()

    c = tf.one_hot(b, 10, on_value=1, off_value=-1)
    print c.eval()
