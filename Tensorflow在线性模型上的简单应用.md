

####Tensorflow在线性模型上的简单应用
```
1.引用库
2.加载一系列的数字图片
3.Tensorflow图构造
  3.1 模型选择
    3.1.1 喂入数据准备
    3.1.2 等待优化的参数
    3.1.3 构造初步的模型
  3.2 等待优化的损失函数
  3.3 创建优化器
  3.4 评价性能
4.Run
  4.1 初始化变量
  4.2 装载数据源
  4.3 开始run训练模型
  4.4 训练之后,对模型进行评价
```
NOTE:以下所有的代码是在Jupyter Notebook中操作
####1.引用库
```
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
```
matplotlib是最著名的Python图表绘制扩展库，它支持输出多种格式的图形图像，并且可以使用多种GUI界面库交互式地显示图表。
使用%matplotlib命令可以将matplotlib的图表直接嵌入到Notebook之中，或者使用指定的界面库显示图表，它有一个参数指定matplotlib图表的显示方式。inline表示将图表嵌入到Notebook中。
其他import语句是用以引入工具库

接下来可以打印出来tensorflow对应的版本号,看看引用是否正常
```
tf.__version__
```
####2.加载一系列的数字图片
MNIST是一个手写数字数据库,tensorflow里已经内置了便捷的加载方式用于加载这些数据
如果指定的目录没有,会联网下载对应文件,然后加载
```
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
```
这里需要需要稍微留意一下，数据是采用One-Hot Encoding
什么是One-Hot Encoding,为什么要采用这种encoding
+ 一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都有他独立的寄存器位，并且在任意时候，其中只有一位有效。
简化就是一串N位数据流（0，1形式的数据流），里边只有一个状态是激活（用1表示），例如 000100
+ 特征之间距离的计算或相似度的计算是非常重要的。这种形式的数据更加合理地描述了俩个没有关联的特征，比如数字1（0000000010），数字7。这样模型在拟合数据之后计算cost更加合理，而我们通过对数据的不断训练来降低cost，更好地拟合数据，获得满意的模型。

接下来，继续，我们可以尝试打印出来这个数据集的size信息
```
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
```
如果你想看看One-Hot Encoding长得什么样子,下面打印出测试数组前面5个对象的数据
```
data.test.labels[0:5, :]
```
 现在这些标记的数据都是One-Hot Encoding,为了后面的运算与显示,我们需要对这些One-Hot Encoding数据使用 数字进行分类
```
 data.test.cls = np.array([label.argmax() for label in data.test.labels])
```
 更加简洁的写法可以这样, 对某一个轴进行压缩,argmax返回的是压缩前最大数的索引
```
 data.test.cls = data.test.labels.argmax(axis=1)
```
 #### 3.Tensorflow图构造
 总算回到问题的核心了,Tensorflow并不是像传统的程序一样,你定义好每一步骤,然后严格按照你这一行行代码执行下去,得出一个结果.更像是一个让你定义一个巧妙的运算图(computational graph),这个运算图我觉得就是一台机器,各个部件协同工作,当各个部件组建完成之后,你就可以喂入数据,让他吐出来结果(这个结果其实就是这台机器觉得他自身模型最适合的参数).
 有一个组件A,说这个地方是专门进行模型的选定,你给我对应的模型参数,我就能做出预测值;
 有一个组件B说它是负责预测值和真实值之间开销的计算方式,当然了,你得给我预测值和真实值;
 有一些组件C则声称只要你设置好开销计算的方式以及对应的学习速度,我就能采用梯度下降找到最低开销对应的参数.
 这个时候,假设一切都很顺利.当这三个组件都配置组装好了之后,喂进去100,000,000条数据,C组件就说它找到了它觉得最优的参数,有这些参数计算出来的模型能以尽可能低的开销拟合真实情况.
 但是有时候你会发现,这个喂入的数据太大了,开销最小计算起来实在是费劲,于是乎我们想了个办法,我一次不喂进去那么多,分成小批量来,这个就是mini-batch,每次喂入10,000条数据.每一次mini-batch的数据运算完成之后,模型参数一般都会发生改变,改变后的模型参数又参与下一次mini-batch数据的计算.
 而每一次mini-batch计算之后,我们都可以打印出来它的learning-cure是什么样,比如说Y轴是准确率,X轴是经过了多少次mini-batch.

 ##### 3.1 模型选择
 这个例子,我们选用一个线性模型来拟合采集到的数据.线性模型的方式 Y_Predict  = X*W + bias.  最终计算真实值Y和Y_Predict之间的开销。不断尝试 优化权重降低开销来fit真实数据。
 ###### 3.1.1 喂入数据准备
既然选择了线性，那我们就需要将样例的数据进行扁平化，每个sample成为一个一维的向量。还记得我们用One-Hot Encoding吧所有的样例标签被映射到一个N维度(这里是10的维度),为了计算Y_Predict和真实值Y之间的cost，Y_Predict也必须是一个10维的向量，W作为权重矩阵，就是[sample_plat_size, 10].
其中X是扁平化了的sample数据,如果是多个sample,那么X就是一个矩阵,进行矩阵运算,来计算出多个预测.
那接下来我们就可以定义数据的一些长度
```
 # We know that MNIST images are 28 pixels in each dimension.
 img_size = 28

 # Images are stored in one-dimensional arrays of this length.
 img_size_flat = img_size * img_size

 # Tuple with height and width of images used to reshape arrays.
 img_shape = (img_size, img_size)

 # Number of classes, one class for each of 10 digits.
 num_classes = 10
```
对于输入主要是俩类，一个是X,一个是Y,由于输入数据是不断喂入的，我们使用placeholder来响应外部动态数据。
```
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])
```
###### 3.1.2 等待优化的参数
经过上面的分析，我们也明白了我们要优化也是我们最终需要的参数是W以及B（bias）

```
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))
```
 ###### 3.1.3 构造初步的模型
```
logits = tf.matmul(x, weights) + biases
```
到了这一步，按照之前说的 logits 应该就是我们得到的预测值。
但是这里我们要做一下处理，归一化。
最优解的寻优过程明显会变得平缓，更容易正确的收敛到最优解。
```
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)
```
##### 3.2 等待优化的损失函数
 softmax_cross_entropy_with_logits传入的是logits,此方法会对数据进行归一化.所以无需传入归一化之后的数据.
```
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
```
计算出来的交叉熵是一个数组，里边每个元素是预测值和真实值之间的交叉熵，我们计算均值来作为参考比较的标准.
```
cost = tf.reduce_mean(cross_entropy)
```
##### 3.3 创建优化器
接下来我们需要为设置我们的优化器,这里我们选择了梯度下降算法,同时需要设置好它要优化的对象(也就是开销)以及对应的学习速率,这里我们先写死采用0.5
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
```
到了这里我们基本上这些组件已经拼装得差不多了,也就是我们基本完成了computational graph的构造.感觉我们就差run起来了.等等,好像还少了什么东西.
也许你已经发现我们还少了一个步骤,就是评价性能.我们的目标是找到一个合适的模型来预测未来,什么样的模型才算好?就需要比较,衡量才能看出来.
 ##### 3.4 评价性能
还记得我们的数据是动态输入的吧.在训练模型阶段,我们采用的是train-set.到了评价阶段,我们的数据源要切换成test-set.这样才能用来评价模型.
否则train-set 扮演俩个角色,运动员和裁判员就不合理了.
```
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
#### 4.Run
创建好了TensorFlow图，需要创建TensorFlow session来运行图。
```
session = tf.Session()
```
#####4.1 初始化变量
我们需要在开始优化weights和biases变量之前对它们进行初始化。
```
session.run(tf.global_variables_initializer())
```
#####4.2装载数据源
还记得我们先前做的喂入数据准备吗?如果忘记了,可以瞄一眼提纲.现在我们需要正式地给我们定义出来的字段load上测试数据.
我们的目标是把数据装载进入x, y_true 这俩个对象中.
假如我们的batch-size 设置为100, 按照我们的如意算盘,多次迭代(每次100个数据)之后获得非常好的参数来拟合我们的测试数据.
```
batch_size = 100
def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
```
还记得我们前面说过,评价模型需要采用test-set,在评价模型的时候,我们把数据源切换到test-set
```
feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}
```
#####4.3 开始run训练模型
```
optimize(num_iterations=1)
```
#####4.4 训练之后,对模型进行评价
我们定义一个方法,使用了测试数据集, 以及我们对accuracy的定义.
```
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
```
定义完成之后,我们就可以打印出来,训练了一次之后,准确度如何
```
print_accuracy()
```
修改num_iterations,再看看accruracy如何.

[完整代码参考](https://github.com/buptsse/TFCodeBook/blob/master/Tensorflow%E5%9C%A8%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B%E4%B8%8A%E7%9A%84%E7%AE%80%E5%8D%95%E5%BA%94%E7%94%A8.ipynb)

Refer:
[数据预处理：独热编码（One-Hot Encoding）](http://blog.csdn.net/pipisorry/article/details/61193868)
[详解numpy的argmax](http://www.cnblogs.com/zhouyang209117/p/6512302.html)
[为什么要对数据进行归一化处理](https://zhuanlan.zhihu.com/p/27627299)
[tf.nn.softmax_cross_entropy_with_logits的用法](http://blog.csdn.net/mao_xiao_feng/article/details/53382790)
[simple-leaner-example](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb)
