# YZM
验证码识别，使用[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/) + Ctc loss实现


## 目标
识别 4 * (26个小写字母 + 26个大写字母 + 10个数字 = 62个字符) 的验证码。

## 运行环境
建议使用anaconda 安装，版本号最好一致。
1. torch == 1.4
2. cpatcha == 0.3
3. tensorboard == 2.1
4. torchvision == 0.5
5. numpy

## 参数设置
default_config.py 中描述了默认参数。
```python
class Config:
    lr = 1e-3 #学习率，假数据集中刚开始使用1e-2会好一点。 但是一般来说是1e-3
    cuda = True # 使用使用gpu
    batch_size = 256
    num_workers = 64 #加载数据集使用的线程数
    print_step = 128 #统计数据，已经保存模型的步
    log_file = "./logs/first_logs/" # tensorboard log 位置
    model_saved = "models_saved/ctc_loss.pth" 
    model_resume = model_saved # 从哪里接着训练模型。
    local_rank = 0  # only for intelliense, useless otherwise 不要动这里 !!!
```

## 数据集

由于现在没有数据集，所以训练的时候，使用captcha库自己生成了一些数据集以供训练。

使用真实的数据集，只需要写自己的dataset，修改FakeDataset构造函数里面的参数。 然后再 captcha_rec.py import 就行。

注意:
1. 假数据集图片的大小为(128, 64)。 如果改变输入大小，所以可能导致模型不能运行，如果比赛数据集图片大小小于假数据集图片，可以resize -> (128,64)。 如果比假数据集图片大，那么最好重新训练。
2. 虽然是62个字符，但是还有一个背景，所以模型识别63个类。建议在预测的时候，把背景的 分数 设置为0，这样就防止识别结果出现背景。


## 运行效果
使用 tensorboard 查看训练结果。1080 大概3个小时可以让模型拟合。
```bash
tensorboard --logdir=logs/first_logs --bind_all
```
不考虑字符位置顺序的准确率为(mean_acc): **0.99**

考虑字符位置的绝对准确率为(abs_acc): **0.96**. 

比赛应该要求的是 需要考虑字符位置的 准确率

## 可改进点
1. 在dataset.py 中，使用了 to_tensor() 把图片转化为 tensor。但是并没有进行归一化和数据增强，因为在假数据集结果已经非常好了。使用比赛的话，最好加上归一化和数据增强，**不然效果会下降很多**
2. 我使用了标准ResNet50，因为假数据集有很多数据，不怕过拟合。如果你们出现过拟合了，可以考虑ResNet修改captcha.py 中的layer
```python
 model = make_model("resnet50", [3, 4, 6, 3], num_classes, label_length, False)
 ```
 其中[3,4,6,3] 就是标准ResNet50 layer 描述，可以做出修改。
 3. 模型在真实数据集，大概应该可以达到0.8以上的效果，如果你们追求0.9+。可以考虑一下弱监督目标检测。
 4. 询问好是否可以使用外部数据源，比如公共数据集 / 模型预训练 / 假数据。 如果可以的话， 用进来。如果不允许使用假数据集，不要从假数据集训练的模型接着训练，即刚刚开始训练的时候，请在run.py 设置 config.model_resume = None
