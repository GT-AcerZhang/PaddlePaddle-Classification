import argparse
import numpy as np
import paddle.fluid as fluid
from utils import utils


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=int,      default="test.jpg")
    parser.add_argument("--model_path", type=int,      default="output/quant_inference_model")
    parser.add_argument("--use_gpu",    type=str2bool, default=True)
    parser.add_argument("--img_size",   type=int,      default=224)
    return parser.parse_args()


# 加载模型
def create_predictor(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    [program, feed_names, fetch_lists] = fluid.io.load_inference_model(dirname=args.model_path,
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


def main():
    args = parse_args()
    operators = create_operators(args)
    exe, program, feed_names, fetch_lists = create_predictor(args)
    data = preprocess(args.image_path, operators)
    data = np.expand_dims(data, axis=0)
    outputs = exe.run(program,
                      feed={feed_names[0]: data},
                      fetch_list=fetch_lists,
                      return_numpy=False)
    lab, porb = postprocess(outputs).__next__()
    print("结果为：%s, 概率为：%f" % (lab, porb))


if __name__ == '__main__':
    main()