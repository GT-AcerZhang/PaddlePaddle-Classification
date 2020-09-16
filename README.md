# PaddlePaddle-Classification
PaddlePaddle实现的图像分类模型

# 安装环境

1. 安装PaddlePaddle 1.8.4 GPU版本
```shell script
python3 -m pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirrors.aliyun.com/pypi/simple/
```

2. 安装依赖环境
```shell script
pip install ujson opencv-python pillow tqdm PyYAML visualdl -i https://mirrors.aliyun.com/pypi/simple/
```

3. 安装PaddleSlim库
```shell script
pip install paddleslim==1.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 安装PaddleClas库
```shell script
pip install ./utils/ppcls-0.0.1-py3-none-any.whl
```

# 训练

1. 根据每个类别的图片放在一个文件夹中，然后把这些文件夹都放在`dataset/images`目录下

2. 执行`create_list.py`生成训练所需的图像列表，同时也会生成标签对应的名称文件`dataset/labels.txt`，图像列表格式如下。
```shell script
dataset/images/class8/test1.jpg 8
dataset/images/class15/test2.jpg 15
dataset/images/class6/test3.jpg 6
```

3. 执行`train.py`开始训练，训练的模型会保存在`output`中，VisualDL的日志保存在`logs`中。执行命令如下，如果想要训练其他的模型，可以更改`config`的配置文件路径。
```shell script
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/ResNet50_vd.yaml
```
