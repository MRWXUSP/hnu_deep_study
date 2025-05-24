# hnu_deep_study
宠物面部识别模型训练与预测

[TOC]

## 1. 快速开始
### 1.1 环境准备
本模型使用Python 3.8版本，建议使用Anaconda创建虚拟环境。
约有如下依赖包：
```bash
pytorch
torchvision
torchaudio
pytorch-cuda
pillow
timm
tqdm
numpy
opencv-python
```
### 1.2 数据准备
请前往[release发布页](https://github.com/MRWXUSP/hnu_deep_study/releases)下载分卷数据集`hnu_deep_study_part1-3.rar`与标签数据集`annotations.rar`，全部解压后应该有如下的目录结构：
```
|-- face_trainset/  提取狗头部特征的训练集
|-- trainset/       原始训练集
|-- valset/     原始验证集
|-- weight/     预训练模型权重
|-- annotations/  训练集标签
```
将他们放在项目根目录下。

### 1.3 训练模型

1. 请确保你的目录结构一定要有以下内容：
    ```
    |-- face_trainset/  提取狗头部特征的训练集
    |-- trainset/       原始训练集
    |-- valset/     原始验证集
    |-- weight/     预训练模型权重
        |-- resnet18/
            |-- model.safetensors
    |-- annotations/  训练集标签
    |-- train.py       训练脚本
    |-- dataset.py    数据集脚本
    |-- model.py      模型脚本
    |-- predict.py   预测脚本
    |-- README.md     本文档
    ```

2. 修改`train.py`中的参数：
    ```python
    gpus = [0,1] #这里填写你想用的GPU编号
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device:', device)
    model.to(device)
    ```
    修改第一行（在文件中约40行左右）中的`gpus`列表，填写你想使用的GPU编号。如果只有一块GPU，则填写为`[0]`。

3. 然后直接运行[train.py](train.py)脚本即可开始训练，训练将会自动开始，会在终端显示训练轮次、损失值等信息。
要特别注意一个打印在终端的参数：`train mae` 。这个值凸显了模型的训练效果，**越小越好**。
据经验，mae将在开始时以`50`左右的值开始下降，当mae无明显变化后，模型训练完成。

#### 训练中断与恢复
训练脚本会自动保存每一轮的模型在`saved_model_age`文件夹中，如果训练中断，可以通过以下方式从上一轮开始继续训练：
修改`train.py`中的`resume`与`last_epoch`参数，约在90行左右：
```python
# 6. 是否恢复模型
resume = 0
last_epoch = 0
if resume and last_epoch > 1:
    model.load_state_dict(torch.load(
        save_model_dir + '/checkpoint_%04d.pth' % (last_epoch),
        map_location=device))
    print('resume ' , save_model_dir + '/checkpoint_%04d.pth' % (last_epoch))
```
其中`resume`设置为`1`表示恢复训练，`last_epoch`设置为上次训练的轮次。
比如文件夹中保存的模型为：`checkpoint_0005.pth`，则`last_epoch`设置为`5`。

### 1.4 预测模型
预测脚本`predict.py`在这里没有什么用处，因为这个题目没有测试集，于是这个脚本只是一个示例。

如果你有自己的测试集，可以修改`predict.py`以下参数：
```python
m_path = 'saved_model_age/checkpoint_0014.pth'
#m_path = 'saved_model_res18_reg/checkpoint_0010.pth'
checkpoint = torch.load(m_path, map_location=device)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
#model.load_state_dict(torch.load(m_path, map_location=device))



# model = model.cuda(device=gpus[0])
model = model.to(device)
model.eval()

files = glob.glob("./testset/*.jpg")
```
修改`m_path`为你想要使用的模型路径，修改`files`为你想要预测的图片路径。

然后直接运行`predict.py`脚本即可开始预测，预测结果将会保存在`./predict_res50_14.txt`文件中。