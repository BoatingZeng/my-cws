修改自：https://github.com/yanshao9798/segmenter

核心结构没有改动，主要是修改了输入输出，还有添加了训练时读取上次模型的功能。

## 用法

### 准备数据

`example_data/pku`目录就是一个工作目录实例。

`char.txt`是字典文件，这个文件是从2014人民日报中提取的，每个字后面的数字是这个字的出现次数。

`pku_train.segd`、`pku_dev.segd`、`pku_test.segd`这个三个分别是训练集、开发集、测试集。

### 训练命令

python3 -m tf_cws.trainer train -p ./example_data/pku -t pku_train.segd -d pku_dev.segd -tb 128 -iter 60 -op adam -lr 0.001 -ld 0.05

#### 参数说明

1. -p：工作目录，训练语料，临时文件，生成的模型文件等等，都会储存到这里
2. -t：训练集
3. -d：开发集
4. -tb：训练时的batch_size，推荐尽可能大一点，太小会训练得很慢
5. -iter：迭代次数，到了这个次数就会自动停止。也可以中途强行停止。程序会在每个周期后，比较分数，如果分数提高，就保存分数和模型。
6. -op：优化算法
7. -lr：初始学习率，通过placeholder传递
8. -ld：学习率下降率，这个是通过自己设定的learning rate schedule来调整学习率，设置为0.0就可以完全由算法决定学习率

### 训练后生成

文件会生成在指定的工作目录里，主要的文件是下面这些：

1. char.txt：这个不是生成的，但是要跟随模型走
2. tags.txt：记录所使用的tag模式
3. trained_model_weights_score：记录分数，用pickle读取，下次再继续训练时，会读取这个作为模型初始分数
4. trained_model_model：模型的一些基本信息，pickle读取
5. checkpoint、trained_model_weights.data-00000-of-00001、trained_model_weights.index：这个几个是tensorflow保存的模型参数。

如果训练时设置了--model参数（模型名字），那上面的部分文件会根据设置的名字改变，具体是：
`<model>_weights_score`、`<model>_model`、`<model>_weights.data-00000-of-00001`、`<model>_weights.index`。因为默认的--model参数是trained_model，所以默认生成的文件名如上面列表所示。


### 测试命令

python3 -m tf_cws.trainer test -p ./example_data/pku -e pku_test.segd

### 实际环境中使用

参照`main.py`