
####Tensorflow上使用简单神经网络
当我们回头去看Tensorflow线性模型的简单应用时候,你会发现,他的模型如下:
<img src="http://images2015.cnblogs.com/blog/140867/201609/140867-20160908142408535-1835352561.png" width="300">
这长得多像神经网络啊....没错,你可以理解这就是一个简化版本的升级网络.
看了一下之前的思路,我们只需要在**模型选择**上动刀子就好.
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



  ####3.1 模型选择
  我们打算使用一个三层的神经网络,其中包含一个输入层(具有784个节点,这是由于图片数据是28*28=784),一个隐藏层(500个节点),输出层(10个节点)
```
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
```
#####3.1.1 喂入数据准备
这个步骤目前来看不需要变更
#####3.1.2 等待优化的参数
我们需要优化的参数是层和层之间的权重以及bias,
所以在 输入层和hidden layer之间会有weight_1, bias1需要关注;hidden layer和输出层之间是weight_2, bias2
但是升级网络的拟合能力实在太强大了,为了防止参数过多,学习过了,出现过拟合的情况,我们对参数进行了正则化.所以我们按照下面的方式来创建权重
 ```
REGULARAZTION_RATE = 0.0001
 def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape,initializer=tf.truncated_normal_initializer(stddev=0.1)) #生成截断正态分布的随机数,标准差为0.1
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights
 ```

#####3.1.3 构造初步的模型
```
#define the forward network
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):#声明第一层神经网络的变量并完成前向传播过程
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)  #tf.nn.relu是作为激活函数

    with tf.variable_scope('layer2'):#声明第二层神经网络的变量并完成前向传播过程
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
```


```
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)  
logits = inference(x, None) #这个地方暂时传入None,不考虑正则化
```

###Refer
[tensorflow中的关键字global_step使用](http://blog.csdn.net/uestc_c2_403/article/details/72403833)
(什么是 L1 L2 正规化 正则化 Regularization (深度学习 deep learning)
)[https://www.youtube.com/watch?v=TmzzQoO8mr4]