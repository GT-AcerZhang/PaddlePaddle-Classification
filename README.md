# PaddlePaddle-Classification
基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 修改的图像分类模型，方便训练自定义数据集和量化训练。

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
pip install ./utils/ppcls-0.0.2-py3-none-any.whl
```

# 训练

1. 根据每个类别的图片放在一个文件夹中，然后把这些文件夹都放在`dataset/images`目录下

2. 执行`create_list.py`生成训练所需的图像列表，同时也会生成标签对应的名称文件`dataset/labels.txt`，图像列表格式如下。
```shell script
dataset/images/class8/test1.jpg 8
dataset/images/class15/test2.jpg 15
dataset/images/class6/test3.jpg 6
```

3. 执行`train.py`开始训练，训练的模型会保存在`output`中，VisualDL的日志保存在`logs`中。执行命令如下，如果想要训练其他的模型，可以更改`config`的配置文件路径，配置文件来源地址：[configs](https://github.com/PaddlePaddle/PaddleClas/tree/master/configs) 特别说明，不支持GoogLeNet。如果训练数据集太少，需要设置num_workers为1,并使用单卡训练。
```shell script
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/ResNet50_vd.yaml
```

# 评估和预测

1. 评估模型的方式是使用预测模型进行一张张图片预测并求准确率的，所以需要导出预测模型，在训练结束之后如果设置了量化训练，就已经自动保存了量化的预测模型，但是普通的预测模型还需要进一步导出，执行以下命令导出预测模型。
```shell script
python export_model.py
```

2. 执行`eval.py`开始评估，包裹普通预测模型和量化预测模型，输出结果如下：
```
start test output/quant_inference_model model accuracy rate...
100%|██████████| 1958/1958 [01:25<00:00, 22.87it/s]
准确率：0.90378, 平均预测时间为：43
======================================================================
start test output/inference_model model accuracy rate...
100%|██████████| 1958/1958 [00:43<00:00, 44.75it/s]
准确率：0.92378, 平均预测时间为：22ms
======================================================================
W0916 15:49:15.037608 14036 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 75, Driver API Version: 10.1, Runtime API Version: 10.0
W0916 15:49:15.047610 14036 device_context.cc:260] device: 0, cuDNN Version: 7.6.
W0916 15:49:23.432540 14036 build_strategy.cc:170] fusion_group is not enabled for Windows/MacOS now, and only effective when running with CUDA GPU.
W0916 15:50:49.659044 14036 build_strategy.cc:170] fusion_group is not enabled for Windows/MacOS now, and only effective when running with CUDA GPU.
```

3. 预测程序分使用量化模型`infer_quant.py`预测和使用普通预测模型`infer.py`预测。