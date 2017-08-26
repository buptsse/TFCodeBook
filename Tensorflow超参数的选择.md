####Tensorflow超参数的选择
从上个例子里,我们写死了learning rate 以及 batch-size,在超参数的选择过程中,还会涉及到其他参数,比如正则项系数,这里我们暂时不考虑其他.只是基于我们上个例子进行讨论.
```
1.learning rate的选择
  1.1 learning rate面临的问题很好理解.
  1.2 如何调整
  1.3 动态变化的learning rate
2.训练停止的时机
  2.1 early-stoping
  2.2 谷底策略
3.MIni-batch size的选择
4.例子实践
  4.1 learning rate
  4.2 mini-batch size
  4.3 谷底策略
```
#####1.learning rate的选择
######1.1 learning rate面临的问题很好理解.
如果太小,梯度下降就太慢了,要花比较久才能到达谷底.但是有利于寻找某个局部最优.
如果太大,又容易迈过了谷底,在俩个山峰之间来回震荡.无法形成收敛.
######1.2 如何调整
在实践中,一般都是通过不断尝试来调整.学习速率设置为0.01，然后观察training cost的走向，如果cost在减小，那你可以逐步地调大学习速率，0.1，1.0….如果cost在增大，那就得减小学习速率，0.001，0.0001….经过一番尝试之后，你可以大概确定学习速率的合适的值.
这个地方你会发现我们并没有使用validation set来进行learning rate的选择,而是使用了training set来评价.
《Neural networks and deep learning》中对这个问题进行了描述.
大意是使用哪种数据集合属于个人偏好.其他超参数主要是为了提高test-set上的accuracy,对accuracy有直接的影响.
而learning rate对于accruracy却是很间接.它的主要目的是控制梯度下降的步伐.
######1.3 动态变化的learning rate
你应该会想到如果在开始阶段,使用某一个learning rate来进行梯度下降,在逼近谷底的时候,不断降低learning rate,这样我们就可以尽可能地逼近谷底.同时又能比较快速下降.想想还是蛮激动的.
#####2.训练停止的时机
######2.1 early-stoping
很合理的想法就是在梯度下降过程中,validation accuracy没法提升的时候,就没有必要再训练下去了.
这种思路合理吗?接下来一俩个迭代没有提升,是不是意味着以后都不会有提升?
正确的做法是，在训练的过程中，记录最佳的validation accuracy，当连续10次epoch（或者更多次）没达到最佳accuracy时，你可以认为“不再提高”.这个策略就叫“no-improvement-in-n”，n即epoch的次数，可以根据实际情况取10、20、30…
毕竟我们的认知是有限的,而未知的世界是无限的,我们只能竭尽我们的视力去瞭望远方,做出判断.
######2.2 谷底策略
结合1.3的动态learning rate和2.1的eraly-stoping,我们在可以构造一个自动的策略,而无需人工干预.
假如对于“no-improvement-in-n”我们设置n=10,在跑了十次之后,validation accuracy还是没有提升,这个时候我们把learning-rate砍掉一半,继续执行,这个思路类似于我们learning rate过大,导致迈过了谷底,在俩个山峰之间震荡,我们通过缩小learning-rate来期望继续往底部走.而当learning-rate变成了原来的1/1024或者1/512我们再进行真正的stop.
#####3.MIni-batch size的选择
size的选择过大过小都是有不足之处
size过大,权重的更新就不会很频繁.导致优化过程漫长
size过小,则无法充分利用矩阵、线性代数库来进行计算的加速,同时小批量的样本数据无法准确体现出整体的cost平均值,容易出现偏离.
整体来看应该根据数据集的规模,硬件能力来选择.这么说还是有点广泛,有一种做法是通过横轴为time,纵轴为 validation accuracy,然后通过多个图,不同图的batch-size不一样来,来观测比较哪个batch-size在同样时间上准确率抬升最快.这里的time说的是计算机的运算时间开销,而不是迭代次数.

#####4.例子实践
######4.1 learning rate
我们通过x轴表示epoch的次数,y轴表示cost
在[上个例子代码](https://github.com/buptsse/TFCodeBook/blob/master/Tensorflow%E5%9C%A8%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B%E4%B8%8A%E7%9A%84%E7%AE%80%E5%8D%95%E5%BA%94%E7%94%A8.ipynb)的基础上我们进行可视化观测
主要思路就是在迭代的过程中,记录下来对应的cost,然后图形化显示.
[具体代码](https://github.com/buptsse/TFCodeBook/blob/master/Tensorflow%E8%B6%85%E5%8F%82%E6%95%B0%E9%80%89%E6%8B%A9-%E9%92%88%E5%AF%B9learningRate.ipynb)
######4.2 mini-batch size
我们通过x轴表示时间开销,y轴表示准确率

######4.3 谷底策略

Refer:
[机器学习算法中如何选取超参数：学习速率、正则项系数、minibatch size](http://blog.csdn.net/u012162613/article/details/44265967)
[【Tensorflow】辅助工具篇——matplotlib介绍（上）](http://blog.csdn.net/mao_xiao_feng/article/details/73430942)
[【Tensorflow】辅助工具篇——matplotlib介绍（下）](http://blog.csdn.net/mao_xiao_feng/article/details/73718388)



