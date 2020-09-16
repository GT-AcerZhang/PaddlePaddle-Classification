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
