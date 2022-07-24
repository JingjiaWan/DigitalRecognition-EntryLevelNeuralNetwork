本文结构：从小到大，慢慢深入壮大代码

### Neural Network类

**Step1: 列框架**

神经网络的过程包括了训练网络以及使用训练后的网络进行预测，为了使代码更健壮，我们不固定输入输出以及隐藏层的节点个数，所以这里大体需要三个函数：

1. 初始化函数：用于设定输入层节点，隐藏层节点以及输出层节点的数量

2. 训练函数：学习给定训练集样本，优化权重

3. 预测函数：给定新的数据输入，网络的输出节点给出答案

   ```python
   # neural network class definition
   class neuralNetwork:
   
       # initialise the neural network
       def __init__(self):
           pass
   
       # train the neural network
       def train(self):
           pass
   
       # query the neural network
       def query(self):
           pass
   ```

   【注】这里使用类的形式来创建神经网络模板，又不固定输入输出节点个数，这样我们就可以创建多个神经网络对象。

**Step2：填充框架**

- ```__init__函数```

  三个作用

  （1）主要是为了输入神经网络的输入输出以及隐藏层，确定神经网络对象的形状

  （2）初始化权重

  权重可以说是网络的核心，我们用权重计算前馈信号，反向传播误差。并且我们所说的训练神经网络其实也是在试图改进网络的**链接权重**。而最开始生成网络时我们的权重是需要随机生成的。设置权重不能太随意，这里直接使用**正态概率分布**采样权重，均值为0，方差为节点传入链接数目的开方。

  （3）除了输入层，其他层的所有节点都需要使用激活函数，为了在改变网络激活函数时不需要改变每一个使用激活函数的地方，我们将激活函数定义到了初始化函数中。lambda起飞函数，一次定义，多次使用。

  ```py
  # initialise the neural network
      def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
          # set number of nodes in each input, hidden, output layer
          self.inodes = inputnodes
          self.hnodes = hiddennodes
          self.onodes = outputnodes
          # learning rate
          self.lr = learningrate
  
          # activation function is the sigmoid function
          self.activation_function = lambda x: scipy.special.expit(x)
          
          # link weights
          # wih -- weight between input and hidden
          # who -- weight between hidden and output
          self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
          self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
  ```

- ```train()函数```

  训练网络，顾名思义，就是训练神经网络的核心步骤了。训练任务分两部分

  第一，针对给定的训练样本计算输出，就是下边的query函数

  第二，将网络输出与样本原本输出进行比较，使用差值反向传播到每一层，然后使用这些差值对网络权重进行更新。

  ​			因为各层结构相似，所以这里拿输出层到隐藏层举例，误差反向传播的公式为：$errors_{hidden} = weight^T_{hidden-output} · errors_{output}$ 输入层到隐藏层之间也类似

  ​			隐藏层第 j 个节点到输出层第 k 个节点之间的连接权重矩阵的更新公式为：$\Delta W_{j,k} = \alpha * E_k * sigmoid(O_k)*(1-sigmoid(O_k)) · O^T_j$ 这里的 * 是对应元素乘法， ·  是矩阵乘法

  【注】1. 这里主要是使用梯度下降算法思想以及反向传播算法推导出权重的更新公式。2. 反向传播算法是基于梯度下降算法的。梯度下降是一个用来寻找最小值的数学上的算法，而神经网络的误差函数也正好是找最小值，所以就使用了梯度下降算法。接着，梯度下降中每次需要下降一定的步长，这个步长就是使用反向传播算法推导出来的值和学习率相乘得来的。关于这两个算法的推导，可以看西瓜书等资料。

  ```python
      def train(self, inputs_list, targets_list):
          # convert inputs list to 2d array
          inputs = numpy.array(inputs_list, ndmin=2).T
          targets = numpy.array(targets_list, ndmin=2).T
  
          # calculate signals into hidden layer
          hidden_inputs = numpy.dot(self.wih, inputs)
          # calculate the signals emerging from hidden layer -- outputs of hidden
          hidden_outputs = self.activation_function(hidden_inputs)
  
          # calculate signals into final output layer
          final_inputs = numpy.dot(self.who, hidden_outputs)
          # calculate the signals emerging from final output layer
          final_outputs = self.activation_function(final_inputs)
  
          # error using BP
          output_errors = targets - final_outputs
          hidden_errors = numpy.dot(self.who.T, output_errors)
          # update the weights between the hidden and output layers
          self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
          # update the weights between the hidden and output layers
          self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
  ```

- ```query()函数```

  ```query()```  函数主要是接受神经元的输入，返回网络的输出。也就是说，如果我们有一个针对某任务训练好的神经网络，我们希望用它来执行一些新的预测任务，我们就可以直接调用```query()```函数得到结果。这个计算过程有两点需要注意：

  （1）我们传入输入层节点的输入信号，需要经过隐藏层，最后从输出层输出

  ​			因为各层结构相似，所以这里拿输入层到隐藏层举例，网络向前传播的公式为：

  ​					隐藏层输入  $X_{hidden} = W_{input-hidden} · I$
  
  ​					隐藏层输出 $O_{hidden} = sigmoid (X_{hidden})$
  
  （2）信号送到隐藏层或输入层节点时，除了传送过程中的权重进行调节，在神经元内部还需要使用一个激活函数进行信号的抑制。本文使用的是 ```sigmoid ``` 激活函数（已在```__init__```函数中选择），此函数在python的 ```scipy``` 库中的名字是 ```expit()```

```python
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer -- outputs of hidden
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outs = self.activation_function(final_inputs)
```

大功告成，接下来就可以使用 Neural Network 类去创建一些对象了。

### 数字识别

- 数据集

  原数据：http://yann.lecun.com/exdb/mnist/，其中包含60000个训练样本，10000个测试样本

  将原数据转换为 CSV 格式的数据：https://pjreddie.com/projects/mnist-in-csv/

- 输入与输出

  就像初始化权重一样，输入输出应该与网络设计以及实际求解问题匹配网络才能更好的工作。

  【注】关于初始化权重以及输入输出的一些小建议：

  1. 输入常见范围是 [0.01, 0.99] 或 [-1, 1]，使用哪个范围，取决于是否匹配问题

     ```py
     # scale inputs in [0.01, 1]
             scaled_inputs = (numpy.asfarray(inputs[1:]) / 255.0 * 0.99) + 0.01
     ```

  2. 输出应该在激活函数能够生成的范围内，例如sigmoid函数，不可能生成小于等于0或大于等于1的值，那我们的输出的合适的范围是 [0.01, 0.99]

     ```py
      # target outputs : all 0.01, except the desired label which is 0.99
             targets = numpy.zeros(output_nodes) + 0.01
             targets[int(inputs[0])] = 0.99
     ```

  3. 初始化权重应该随机，值较小，避免零值

- 训练与测试

  ```python
  	# train network
      inputs_list = loadData("../dataset/mnist_train_100.csv")
      for record in inputs_list:
          # split the record by the ',' commas : str -> list of str
          inputs = record.split(',')
          # scale inputs in [0.01, 1]
          scaled_inputs = (numpy.asfarray(inputs[1:]) / 255.0 * 0.99) + 0.01
          # target outputs : all 0.01, except the desired label which is 0.99
          targets = numpy.zeros(output_nodes) + 0.01
          targets[int(inputs[0])] = 0.99
          DigitalRec.train(scaled_inputs, targets)
  
      # test network
      tests_list = loadData("../dataset//mnist_test_10.csv")
      # scordcard for how well the network perform, initially empty
      scordcard = []
      for record in tests_list:
          inputs = record.split(',')
          scaled_inputs = (numpy.asfarray(inputs[1:]) / 255.0 * 0.99) + 0.01
          sample_output = int(inputs[0])
          outputs = DigitalRec.query(scaled_inputs)
          # the index of the highest value corresponds to the network_output
          network_output = numpy.argmax(outputs)
          if(sample_output == network_output):
              scordcard.append(1)
          else:
              scordcard.append(0)
  ```

  

到这基本上已经结束了，但是为了训练出更好的网络我们还应该进行一些改进，也就是我们常常说的**调参**。接下来就开始炼丹了.......

### 调参

- 学习率

  直观上来说，在梯度下降过程中，学习率太大会导致权重反复横跳与超调，学习率太小又会导致训练时间过长，所以我们应该找出使模型训练效果更好以及训练时间可接受的学习率。下边实验了十个学习率并将性能绘制成了折线图。

  ![learning rate](D:\Projects\DigitalRecognition-EntryLevelNeuralNetwork\performance\learning rate.png)

  所以此网络选择 学习率 为0.2最合适，此时花费时间最少，准确率还相对较高。

- 隐藏层节点个数

  隐藏层将输入转变为输出，是学习发生的场所，更细致的说，是隐藏层节点前后的链接权重具有学习能力。如果隐藏层节点太少，将会导致网络学习能力不足，如果节点太多，又会导致网络难以训练。我们必须选择一个可容忍的运行时间内的某个数量。下边实验了十个不同隐藏层节点个数并绘制成了折线图。

  ![hidden_nodes](D:\Projects\DigitalRecognition-EntryLevelNeuralNetwork\performance\hidden_nodes.png)

350时的性能最好，但训练时间也长，退而求其次，节点个数为200时性能也不差并且训练时间一分钟也可以接受

【注】上述训练结果并不太好，出现了两次峰值，个数为300时突然降低有可能是因为学习率或其他原因造成的。这样的就需要我们调整学习率再次进行实验。

- epoch

  epoch是指多次使用数据集进行重复的训练。这样做的好处是给数据更多梯度下降的机会，这样有助于权重更新，效果可能会更好，但也要防止过拟合。

  ![epoch](D:\Projects\DigitalRecognition-EntryLevelNeuralNetwork\performance\epoch.png)

  可以看到，大概epoch为11次的时候性能可以达到最大

最后，调参是一个漫长的过程，以上只是简单列出了几个指标，我们还可以通过改变激活函数以及尝试给网络加层数进行训练，除此之外，参数的影响并不是单一的，我们应该组合这些参数指标继续训练，反正最终目的是希望得到一个泛化能力强的网络。
