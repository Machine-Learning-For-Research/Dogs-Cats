# Dogs-Cats

*猫狗二分类图片识别*

#### 模型结构

两个卷积层 + 两个FC层

```python
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
```

