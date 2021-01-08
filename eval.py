import argparse
import time
import numpy as np
import paddle.fluid as fluid
from utils import utils
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu",  type=bool,     default=True)
    parser.add_argument("--img_size", type=int,      default=224)
    return parser.parse_args()


def evaluate_quant(data_list, model_path):
    # 加载模型
    def create_predictor(args):
        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        [program, feed_names, fetch_lists] = fluid.io.load_inference_model(dirname=model_path,
                                                                           executor=exe,
                                                                           model_filename='__model__',
                                                                           params_filename='__params__')
        compiled_program = fluid.compiler.CompiledProgram(program)

        return exe, compiled_program, feed_names, fetch_lists

    # 获取预处理op
    def create_operators(args):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0

        decode_op = utils.DecodeImage()
        resize_op = utils.ResizeImage(resize_short=256)
        crop_op = utils.CropImage(size=(args.img_size, args.img_size))
        normalize_op = utils.NormalizeImage(scale=img_scale, mean=img_mean, std=img_std)
        totensor_op = utils.ToTensor()

        return [decode_op, resize_op, crop_op, normalize_op, totensor_op]

    # 执行预处理
    def preprocess(fname, ops):
        data = open(fname, 'rb').read()
        for op in ops:
            data = op(data)
        return data

    # 提取预测结果
    def postprocess(outputs, topk=5):
        output = outputs[0]
        prob = np.array(output).flatten()
        index = prob.argsort(axis=0)[-topk:][::-1].astype('int32')
        return zip(index, prob[index])

    args = parse_args()
    operators = create_operators(args)
    exe, program, feed_names, fetch_lists = create_predictor(args)

    # 开始预测评估
    with open(data_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    results = []
    print('start test %s model accuracy rate...' % model_path)
    start = time.time()
    for line in tqdm(lines):
        path, id = line.replace('\n', '').split(' ')
        data = preprocess(path, operators)
        data = np.expand_dims(data, axis=0)
        outputs = exe.run(program,
                          feed={feed_names[0]: data},
                          fetch_list=fetch_lists,
                          return_numpy=False)
        lab, porb = postprocess(outputs).__next__()
        if lab == int(id):
            results.append(1)
    end = time.time()
    t = int(round((end - start) * 1000)) / len(lines)
    print("准确率：%0.5f, 平均预测时间为：%d" % (sum(results) / len(lines), t))
    print('=' * 70)


def evaluate_infer(data_list, model_path):
    # 加载模型
    def create_predictor(args):
        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        [program, feed_names, fetch_lists] = fluid.io.load_inference_model(dirname=model_path,
                                                                           executor=exe,
                                                                           model_filename='__model__',
                                                                           params_filename='__params__')
        compiled_program = fluid.compiler.CompiledProgram(program)

        return exe, compiled_program, feed_names, fetch_lists

    # 获取预处理op
    def create_operators(args):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0

        decode_op = utils.DecodeImage()
        resize_op = utils.ResizeImage(resize_short=256)
        crop_op = utils.CropImage(size=(args.img_size, args.img_size))
        normalize_op = utils.NormalizeImage(scale=img_scale, mean=img_mean, std=img_std)
        totensor_op = utils.ToTensor()

        return [decode_op, resize_op, crop_op, normalize_op, totensor_op]

    # 执行预处理
    def preprocess(fname, ops):
        data = open(fname, 'rb').read()
        for op in ops:
            data = op(data)
        return data

    # 提取预测结果
    def postprocess(outputs, topk=5):
        output = outputs[0]
        prob = np.array(output).flatten()
        index = prob.argsort(axis=0)[-topk:][::-1].astype('int32')
        return zip(index, prob[index])

    args = parse_args()
    operators = create_operators(args)
    exe, program, feed_names, fetch_lists = create_predictor(args)

    # 开始预测评估
    with open(data_list, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    results = []
    print('start test %s model accuracy rate...' % model_path)
    start = time.time()
    for line in tqdm(lines):
        path, id = line.replace('\n', '').split(' ')
        data = preprocess(path, operators)
        data = np.expand_dims(data, axis=0)
        outputs = exe.run(program,
                          feed={feed_names[0]: data},
                          fetch_list=fetch_lists,
                          return_numpy=False)
        lab, porb = postprocess(outputs).__next__()
        if lab == int(id):
            results.append(1)
    end = time.time()
    t = int(round((end - start) * 1000)) / len(lines)
    print("准确率：%0.5f, 平均预测时间为：%dms" % (sum(results) / len(lines), t))
    print('=' * 70)


if __name__ == '__main__':
    evaluate_quant('dataset/test_list.txt', 'output/quant_inference_model')
    evaluate_infer('dataset/test_list.txt', 'output/inference_model')
