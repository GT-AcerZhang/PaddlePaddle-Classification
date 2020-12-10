import argparse
import os
import shutil
import sys

import numpy as np
import paddle.fluid as fluid
from visualdl import LogWriter
from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils.save_load import init_model, save_model
from ppcls.utils import logger
from utils import program
import paddleslim as slim


def parse_args():
    parser = argparse.ArgumentParser("PaddleClas train script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/ResNet50_vd.yaml',
                        help='config file path')
    parser.add_argument('--vdl_dir',
                        type=str,
                        default="logs",
                        help='VisualDL logging directory for image.')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('--use_quant',
                        type=bool,
                        default=True,
                        help='If use slim quant train.')
    parser.add_argument("--output_path",
                        type=str,
                        default='output/quant_inference_model/',
                        help='Save quant models.')
    args = parser.parse_args()
    return args


def main(args):
    config = get_config(args.config, overrides=args.override, show=True)
    # 如果需要量化训练，就必须开启评估
    if not config.validate and args.use_quant:
        logger.error("=====>Train quant model must use validate!")
        sys.exit(1)
    if config.epochs < 6 and args.use_quant:
        logger.error("=====>Train quant model epochs must greater than 6!")
        sys.exit(1)
    # 设置是否使用 GPU
    use_gpu = config.get("use_gpu", True)
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    best_top1_acc = 0.0

    # 获取训练数据和模型输出
    if not config.get('use_ema'):
        train_dataloader, train_fetchs, out, softmax_out = program.build(config,
                                                                         train_prog,
                                                                         startup_prog,
                                                                         is_train=True,
                                                                         is_distributed=False)
    else:
        train_dataloader, train_fetchs, ema, out, softmax_out = program.build(config,
                                                                              train_prog,
                                                                              startup_prog,
                                                                              is_train=True,
                                                                              is_distributed=False)
    # 获取评估数据和模型输出
    if config.validate:
        valid_prog = fluid.Program()
        valid_dataloader, valid_fetchs, _, _ = program.build(config,
                                                             valid_prog,
                                                             startup_prog,
                                                             is_train=False,
                                                             is_distributed=False)
        # 获取训练集的valid格式数据
        if config.validate_train is not None and config.validate_train:
            valid_dataloader_tain, valid_fetchs, _, _ = program.build(config,
                                                                      valid_prog,
                                                                      startup_prog,
                                                                      is_train=False,
                                                                      is_distributed=False)
        # 克隆评估程序，可以去掉与评估无关的计算
        valid_prog = valid_prog.clone(for_test=True)

    # 创建执行器
    exe = fluid.Executor(places[0])
    exe.run(startup_prog)

    # 加载模型，可以是预训练模型，也可以是检查点
    init_model(config, train_prog, exe)

    train_reader = Reader(config, 'train')()
    train_dataloader.set_sample_list_generator(train_reader, places)

    compiled_train_prog = program.compile(config, train_prog, train_fetchs['loss'][0].name)

    if config.validate:
        valid_reader = Reader(config, 'valid')()
        valid_dataloader.set_sample_list_generator(valid_reader, places)
        # 获取训练集的valid格式数据
        if config.validate_train is not None and config.validate_train:
            valid_reader_train = Reader(config, 'valid1')()
            valid_dataloader_tain.set_sample_list_generator(valid_reader_train, places)
        compiled_valid_prog = program.compile(config, valid_prog, share_prog=compiled_train_prog)

    vdl_writer = LogWriter(args.vdl_dir)

    for epoch_id in range(config.epochs - 5):
        # 训练一轮
        program.run(train_dataloader, exe, compiled_train_prog, train_fetchs, epoch_id, 'train', config, vdl_writer)

        # 执行一次评估
        if config.validate and epoch_id % config.valid_interval == 0:
            if config.get('use_ema'):
                logger.info(logger.coloring("EMA validate start..."))
                with ema.apply(exe):
                    _ = program.run(valid_dataloader, exe,
                                    compiled_valid_prog, valid_fetchs,
                                    epoch_id, 'valid', config)
                logger.info(logger.coloring("EMA validate over!"))

            top1_acc = program.run(valid_dataloader, exe, compiled_valid_prog, valid_fetchs, epoch_id, 'valid', config)
            # 计算训练集的准确率
            if config.validate_train is not None and config.validate_train:
                train_top1_acc = program.run(valid_dataloader_tain, exe, compiled_valid_prog, valid_fetchs, epoch_id,
                                             'valid', config)
            if vdl_writer:
                print('=============', top1_acc, train_top1_acc)
                logger.scaler('valid_avg', top1_acc, epoch_id, vdl_writer)
                # 保存训练集的准确率
                if config.validate_train is not None and config.validate_train:
                    logger.scaler('train_avg', train_top1_acc, epoch_id, vdl_writer)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                message = "The best top1 acc {:.5f}, in epoch: {:d}".format(best_top1_acc, epoch_id)
                logger.info("{:s}".format(logger.coloring(message, "RED")))
                if epoch_id % config.save_interval == 0:
                    model_path = os.path.join(config.model_save_dir, config.ARCHITECTURE["name"])
                    save_model(train_prog, model_path, "best_model")

        # 保存模型
        if epoch_id % config.save_interval == 0:
            model_path = os.path.join(config.model_save_dir, config.ARCHITECTURE["name"])
            if epoch_id >= 3 and os.path.exists(os.path.join(model_path, str(epoch_id - 3))):
                shutil.rmtree(os.path.join(model_path, str(epoch_id - 3)), ignore_errors=True)
            save_model(train_prog, model_path, epoch_id)

    # 量化训练
    if args.use_quant and config.validate:
        # 执行量化训练
        quant_program = slim.quant.quant_aware(train_prog, exe.place, for_test=False)
        # 评估量化的结果
        val_quant_program = slim.quant.quant_aware(valid_prog, exe.place, for_test=True)

        fetch_list = [f[0] for f in train_fetchs.values()]
        metric_list = [f[1] for f in train_fetchs.values()]
        for i in range(5):
            for idx, batch in enumerate(train_dataloader()):
                metrics = exe.run(program=quant_program, feed=batch, fetch_list=fetch_list)
                for i, m in enumerate(metrics):
                    metric_list[i].update(np.mean(m), len(batch[0]))
                fetchs_str = ''.join([str(m.value) + ' ' for m in metric_list])

                if idx % 10 == 0:
                    logger.info("quant train : " + fetchs_str)

        fetch_list = [f[0] for f in valid_fetchs.values()]
        metric_list = [f[1] for f in valid_fetchs.values()]
        for idx, batch in enumerate(valid_dataloader()):
            metrics = exe.run(program=val_quant_program, feed=batch, fetch_list=fetch_list)
            for i, m in enumerate(metrics):
                metric_list[i].update(np.mean(m), len(batch[0]))
            fetchs_str = ''.join([str(m.value) + ' ' for m in metric_list])

            if idx % 10 == 0:
                logger.info("quant valid: " + fetchs_str)

        # 保存量化训练模型
        float_prog, int8_prog = slim.quant.convert(val_quant_program, exe.place, save_int8=True)
        fluid.io.save_inference_model(dirname=args.output_path,
                                      feeded_var_names=['feed_image'],
                                      target_vars=[softmax_out],
                                      executor=exe,
                                      main_program=float_prog,
                                      model_filename='__model__',
                                      params_filename='__params__')


if __name__ == '__main__':
    args = parse_args()
    main(args)
